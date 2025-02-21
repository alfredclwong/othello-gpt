# %%
import itertools
from itertools import product
from pathlib import Path

import einops
import huggingface_hub as hf
import pandas as pd
import plotly.graph_objects as go
import torch as t
import wandb
from datasets import load_dataset
from jaxtyping import Float
from plotly.subplots import make_subplots
from tqdm import tqdm

from othello_gpt.data.vis import plot_game, plot_probe_preds
from othello_gpt.research.targets import (
    captures_target,
    flip_dir_target,
    legality_target,
    next_move_target,
    prev_tem_target,
    theirs_empty_mine_target,
)
from othello_gpt.util import get_all_squares, load_model, load_probes, test_linear_probe

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"
probe_dir.mkdir(parents=True, exist_ok=True)

hf.login((root_dir / "secret.txt").read_text())
wandb.login()

size = 6
all_squares = get_all_squares(size)
dataset_dict = load_dataset("awonga/othello-gpt")

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# %%
model = load_model(device, "awonga/othello-gpt-7M")

# %%
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    combos=["t+m", "t-m", "t-e", "t-pt", "m-pm"],
    normed=False,
)
probes_normed = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    combos=["t+m", "t-m", "t-e", "t-pt", "m-pm"],
)
{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
ps = probes_normed

linear_probe_tem = t.stack([ps[k] for k in "tem"], dim=-2)
linear_probe_cap = t.stack([-ps["c"], ps["c"]], dim=-2)
linear_probe_legal = t.stack([-ps["l"], ps["l"]], dim=-2)
linear_probe_ptem = t.stack([ps["p" + k] for k in "tem"], dim=-2)
linear_probe_pptem = t.stack([ps["pp" + k] for k in "tem"], dim=-2)
linear_probe_dir = t.stack([ps["d"], -ps["d"]], dim=-2)
linear_probe_unembed = t.stack([-ps["u"], ps["u"]], dim=-2)

# %%
pptem_target = lambda x, device: prev_tem_target(x, device, n_shift=2)
probe_targets = [
    ("unembed (legal)", linear_probe_unembed, legality_target),
    ("unembed (next move)", linear_probe_unembed, next_move_target),
    ("tem", linear_probe_tem, theirs_empty_mine_target),
    ("cap", linear_probe_cap, captures_target),
    ("legal (legal)", linear_probe_legal, legality_target),
    ("legal (next move)", linear_probe_legal, next_move_target),
    ("ptem", linear_probe_ptem, prev_tem_target),
    ("pptem", linear_probe_pptem, pptem_target),
    ("dir", linear_probe_dir, flip_dir_target),
]

test_dataset = dataset_dict["test"].take(1000)

probe_accuracies = {}
probe_losses = {}
for name, probe, target_fn in tqdm(probe_targets):
    test_y: Float[t.Tensor, "n_test pos n_out"] = target_fn(test_dataset, device)
    test_loss, test_accs = test_linear_probe(
        model,
        device,
        test_dataset,
        test_y,
        probe,
        target_fn=lambda x: target_fn(x, device),
        scalar_loss=False,
    )

    test_loss = einops.reduce(
        test_loss, "layer batch pos n_out -> layer n_out", t.nanmean
    )
    test_accs = einops.reduce(
        test_accs.float(), "layer batch pos n_out -> layer n_out", t.nanmean
    )

    probe_accuracies[name] = test_accs
    probe_losses[name] = test_loss.detach().cpu()

# %%
colours = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#800000",
    "#008000",
    "#000080",
    "#808000",
    "#800080",
    "#008080",
    "#FF8080",
    "#80FF80",
    "#8080FF",
    "#FFFF80",
    "#FF80FF",
    "#80FFFF",
    "#C0C0C0",
    "#404040",
]
cmap = {
    name: colour for name, colour in zip(probe_accuracies, itertools.cycle(colours))
}

fig_loss_layer = make_subplots(
    rows=1, cols=2, subplot_titles=("Test Loss", "Test Accuracy")
)

for name in probe_losses:
    fig_loss_layer.add_trace(
        go.Scatter(
            x=list(range(probe_losses[name].shape[0])),
            y=probe_losses[name].mean(dim=-1),
            mode="lines",
            name=name,
            line=dict(color=cmap[name]),
            legendgroup=name,
        ),
        row=1,
        col=1,
    )
    fig_loss_layer.add_trace(
        go.Scatter(
            x=list(range(probe_accuracies[name].shape[0])),
            y=probe_accuracies[name].mean(dim=-1),
            mode="lines",
            name=name,
            line=dict(color=cmap[name]),
            showlegend=False,
            legendgroup=name,
        ),
        row=1,
        col=2,
    )

fig_loss_layer.update_layout(
    height=600, width=1200, title_text="Test Loss and Accuracy Across Layers"
)
fig_loss_layer.show()

# %%
fig_loss = make_subplots(rows=size, cols=size, y_title="loss", x_title="layer")
fig_accs = make_subplots(rows=size, cols=size, y_title="acc", x_title="layer")

for y in range(size):
    for x in range(size):
        showlegend = y == 0 and x == 0
        for name in probe_losses:
            if name == "dir":
                continue
            middle_indices = [size // 2 - 1, size // 2]
            if "unembed" in name and y in middle_indices and x in middle_indices:
                continue
            fig_loss.add_trace(
                go.Scatter(
                    y=probe_losses[name][:, y * size + x],
                    mode="lines",
                    name=name,
                    line=dict(color=cmap[name]),
                    showlegend=showlegend,
                    legendgroup=name,
                ),
                row=y + 1,
                col=x + 1,
            )
            fig_accs.add_trace(
                go.Scatter(
                    y=probe_accuracies[name][:, y * size + x],
                    mode="lines",
                    name=name,
                    line=dict(color=cmap[name]),
                    showlegend=showlegend,
                    legendgroup=name,
                ),
                row=y + 1,
                col=x + 1,
            )
fig_size = 1200
fig_loss.update_layout(
    height=fig_size, width=fig_size, title_text="Test Loss for Each Square"
)
fig_accs.update_layout(
    height=fig_size, width=fig_size, title_text="Test Accuracy for Each Square"
)
fig_accs.update_yaxes(range=[0.5, 1], row="all", col="all")

fig_loss.show()
fig_accs.show()

# Why is loss lower and acc higher for legal vs unembed on same legality target?
# Legal probe minimises cross-entropy for whether a specific square is legal, given a residual vector
# Unembed (transformer) mininmises cross-entropy over the next token

# %%
batch = dataset_dict["test"].take(1)
plot_game(batch[0])
preds = [
    (linear_probe_unembed, legality_target, "Unembed 'probe'", 60),  # TODO test_loss.argmax()
    (linear_probe_legal, legality_target, "Legal probe", 60),
    (linear_probe_tem, theirs_empty_mine_target, "TEM probe", 51),
    (linear_probe_ptem, prev_tem_target, "PTEM probe", 51),
    (linear_probe_pptem, pptem_target, "PPTEM probe", 51),
    (linear_probe_cap, captures_target, "Captures probe", 50),
    (linear_probe_dir, flip_dir_target, "Directions probe", 34),
]
for probe, target_fn, title, probe_layer in preds:
    plot_probe_preds(
        model,
        device,
        probe,
        batch,
        target_fn=lambda x: target_fn(x, device),
        layer=probe_layer,
        index=0,
        title=title,
    )

# %%
# Visualise cross-orthogonality between linear probes, across layers and features
dot_names, dot_probes = zip(
    *{k: v for k, v in probes_normed.items() if k != "d"}.items()
)
dot_probes = t.stack(tuple(dot_probes), dim=-1)
probe_layers = [0, 50, 60]
dot_probes = dot_probes[..., probe_layers, :]
print(dot_probes.shape)
dots = einops.einsum(
    dot_probes,
    dot_probes,
    "d_model n_out layer_0 probe_0, d_model n_out layer_1 probe_1 -> n_out probe_0 probe_1 layer_0 layer_1",
)
# probe_layers = list(range(dot_probes.shape[-2]))
dots = einops.reduce(
    dots,
    "n_out probe_0 probe_1 layer_0 layer_1 -> (layer_0 probe_0) (layer_1 probe_1)",
    t.nanmean,
)
dots = t.tril(dots)

index = [f"L{l} {probe_name}" for l, probe_name in product(probe_layers, dot_names)]
dots_df = pd.DataFrame(dots.cpu(), index=index, columns=index)

fig = go.Figure(
    data=go.Heatmap(
        z=dots_df.values,
        x=dots_df.columns,
        y=dots_df.index,
        colorscale="RdBu",
        zmid=0,
    ),
)

fig.update_layout(
    title="Heatmap of Dot Products Between Probes",
    height=720,
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(len(index))),
        ticktext=dots_df.columns,
        tickfont=dict(size=8),
        tickangle=30,
    ),
    yaxis=dict(
        tickmode="array",
        tickvals=list(range(len(index))),
        ticktext=dots_df.index,
        tickfont=dict(size=8),
    ),
    shapes=[
        dict(
            type="line",
            x0=i * len(dot_names) - 0.5,
            y0=-0.5,
            x1=i * len(dot_names) - 0.5,
            y1=len(index) - 0.5,
            line=dict(color="black", width=1),
        )
        for i in range(1, len(probe_layers))
    ]
    + [
        dict(
            type="line",
            x0=-0.5,
            y0=i * len(dot_names) - 0.5,
            x1=len(index) - 0.5,
            y1=i * len(dot_names) - 0.5,
            line=dict(color="black", width=1),
        )
        for i in range(1, len(probe_layers))
    ],
)

fig.show()

# %%
# Visualise orthogonality between feature probes for different squares
board_probes = [
    linear_probe_tem,
    linear_probe_ptem,
    linear_probe_pptem,
    linear_probe_cap,
    linear_probe_legal,
]
board_probes = t.cat([p[:, :, :, 1:-1] for p in board_probes], dim=-2)
board_probes /= board_probes.norm(dim=0, keepdim=True)
positional_dots = einops.einsum(
    board_probes[:, : board_probes.shape[1] // 2],
    board_probes,
    "d_model rc0 n_probe n_layer, d_model rc1 n_probe n_layer -> rc0 rc1 n_probe n_layer",
)
# positional_dots[*(range(size) for _ in range(4))] = 0
positional_dots = einops.reduce(
    positional_dots, "rc0 rc1 n_probe n_layer -> (rc0 n_probe) rc1", "mean"
).cpu()
positional_dots = einops.rearrange(positional_dots, "n (r c) -> n r c", r=size)
positional_names = ["t", "e", "m", "pt", "pe", "pm", "ppt", "ppe", "ppm", "c", "nc", "l", "nl"]
plot_game(
    {"boards": positional_dots},
    hovertext=positional_dots,
    reversed=False,
    subplot_titles=[
        f"{chr(ord('A') + x)}{y + 1} {p}"
        for y in range(size // 2)
        for x in range(size)
        for p in positional_names
    ],
    shift_legalities=False,
    n_cols=len(positional_names),
)

# %%
# No need for model to learn mine/theirs for last move because it's always the empty square! (because we filtered out pass games)

## ANALYSE NEURON ACTIVATIONS
# Identify direct circuits (direct logit attribution) e.g. neuron out -> W_out -> W_U
# Identify max activating datasets
# Identify statistically interesting neurons
# Decompompose entire logit components?
# Analyse neuron specificity (spectrum plots, maybe identify polysemanticity?)

# Probe flipped pieces?

# Is mapping a 128 dim space to a 6x6 board impressive?
# Can we construct a NN that does the same job?

# %%
# Find out how H0 constructs an 89% accurate board state
# Hypothesis: each token tracks the tiles that it captured when the move was played
# After H0, we have [my moves; their moves; my moves flipped; their moves flipped]
# Linear probe can then +- to get the board state

# %%
