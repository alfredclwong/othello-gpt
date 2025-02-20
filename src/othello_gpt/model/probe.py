# %%
import datetime as dt
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import einops
import huggingface_hub as hf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch as t
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from jaxtyping import Float

from othello_gpt.data.vis import plot_game, plot_probe_preds
from othello_gpt.research.targets import (
    captures_target,
    forward_probe,
    legality_target,
    # original_colour_target,
    theirs_empty_mine_target,
    # flip_parity_target,
    # mine_flip_target,
    # omine_flip_target,
    prev_tem_target,
    tem_captures_target,
)
from othello_gpt.util import get_all_squares, load_model, load_probes

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
## TRAIN LINEAR PROBES
# A linear probe maps residual vectors (n_batch, d_model) to e.g. board representations (n_batch, size, size)
# This helps us to discover interpretable directions in activation space

# Key concepts:
#  - training linear probes
#  - causal interventions
#  -


@dataclass
class LinearProbeTrainingArgs:
    n_epochs: int = 6
    lr: float = 1e-3
    batch_size: int = 256
    n_steps_per_epoch: int = 200
    n_test: int = 1000
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-3
    use_wandb: bool = True
    wandb_project: str | None = "othello-gpt-probe"
    wandb_name: str | None = None
    warmup_steps: int = 100


def test_linear_probe(test_dataset, test_y, linear_probe, target_fn):
    with t.inference_mode():
        test_y_pred, test_loss = forward_probe(
            model, device, linear_probe, test_dataset, target_fn
        )
    test_accs = (test_y_pred > np.log(0.5)).argmax(-1) == test_y  # TODO check
    test_accs = einops.reduce(
        test_accs.float(), "layer batch pos row col -> layer", "mean"
    )
    test_accs = test_accs.cpu()  # .round(decimals=4)
    return test_loss, test_accs


def train_linear_probe(
    model: HookedTransformer,
    args: LinearProbeTrainingArgs,
    target_fn: Callable,
):
    test_dataset = dataset_dict["test"].take(args.n_test)
    test_y: Float[t.Tensor, "n_test pos row col"] = target_fn(test_dataset).to(device)
    d_probe = test_y.max().item() + 1
    n_probes = model.cfg.n_layers * 2 + 1

    linear_probe = t.randn(
        (model.cfg.d_model, size, size, d_probe, n_probes)
    ) / np.sqrt(model.cfg.d_model)
    linear_probe = linear_probe.to(device)
    linear_probe.requires_grad = True
    print(f"{linear_probe.shape=}")

    test_loss, test_accs = test_linear_probe(
        test_dataset, test_y, linear_probe, target_fn
    )

    batch_indices = t.randint(
        0,
        len(dataset_dict["train"]),
        (args.n_epochs, args.n_steps_per_epoch, args.batch_size),
    )

    cols = ["input_ids", "boards", "coords", "legalities", "flips", "originals"]
    train_dataset = dataset_dict["train"].select_columns(cols)

    optimizer = t.optim.AdamW(
        [linear_probe], lr=args.lr, betas=args.betas, weight_decay=args.weight_decay
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

    pbar = tqdm(total=args.n_steps_per_epoch * args.n_epochs)
    step = 0
    for i in range(args.n_epochs):
        for j in range(args.n_steps_per_epoch):
            # TODO enable pin_memory
            batch = train_dataset.select(batch_indices[i, j, :])
            _, loss = forward_probe(model, device, linear_probe, batch, target_fn)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.update()
            pbar.set_description(
                f"Epoch {i + 1}/{args.n_epochs} {loss=:.4f} {test_accs.mean()=}"
            )
            if args.use_wandb and step >= args.warmup_steps:
                wandb.log({"train_loss": loss}, step=step)
            step += 1

        test_loss, test_accs = test_linear_probe(
            test_dataset, test_y, linear_probe, target_fn
        )

        if args.use_wandb:
            wandb.log({"eval_loss": test_loss}, step=step)
            wandb.log(
                {f"eval_acc_{i}": test_accs[i].item() for i in range(n_probes)},
                step=step,
            )

    if args.use_wandb:
        wandb.finish()

    print(test_accs)

    return linear_probe

# %%
test_dataset = dataset_dict["test"].take(10)
captures_target(test_dataset, device).shape

# %%
args = LinearProbeTrainingArgs()
# args = LinearProbeTrainingArgs(
#     use_wandb=False, n_epochs=2, n_steps_per_epoch=10, lr=1e-3
# )

target_fn = lambda x: prev_tem_target(x, device)
# linear_probe = train_linear_probe(model, args, target_fn)
# t.save(
#     linear_probe,
#     probe_dir / f"linear_probe_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_ptem_7M.pt",
# )

# tem_7M accs
# tensor([0.3304, 0.7273, 0.8151, 0.8190, 0.8202, 0.8249, 0.8255, 0.8402, 0.8406,
#         0.8447, 0.8451, 0.8489, 0.8489, 0.8679, 0.8683, 0.8728, 0.8725, 0.8945,
#         0.8947, 0.8988, 0.8988, 0.9013, 0.9014, 0.9032, 0.9030, 0.9117, 0.9120,
#         0.9190, 0.9198, 0.9288, 0.9297, 0.9317, 0.9322, 0.9367, 0.9371, 0.9420,
#         0.9421, 0.9522, 0.9519, 0.9559, 0.9563, 0.9639, 0.9643, 0.9688, 0.9691,
#         0.9718, 0.9719, 0.9749, 0.9748, 0.9775, 0.9775, 0.9809, 0.9803, 0.9810,
#         0.9778, 0.9776, 0.9752, 0.9747, 0.9745, 0.9745, 0.9743])

# tensor([0.3609, 0.7463, 0.8087, 0.8233, 0.8249, 0.8299, 0.8299, 0.8380, 0.8375,
#         0.8415, 0.8412, 0.8470, 0.8471, 0.8539, 0.8536, 0.8610, 0.8604, 0.8854,
#         0.8850, 0.8916, 0.8920, 0.8958, 0.8963, 0.9008, 0.9012, 0.9106, 0.9112,
#         0.9135, 0.9142, 0.9205, 0.9214, 0.9299, 0.9301, 0.9368, 0.9375, 0.9438,
#         0.9434, 0.9498, 0.9496, 0.9556, 0.9555, 0.9591, 0.9590, 0.9622, 0.9619,
#         0.9669, 0.9674, 0.9732, 0.9732, 0.9785, 0.9785, 0.9811, 0.9802, 0.9812,
#         0.9791, 0.9794, 0.9761, 0.9755, 0.9750, 0.9746, 0.9744])

# cap_7M accs
# tensor([0.7442, 0.8991, 0.9620, 0.9623, 0.9635, 0.9636, 0.9642, 0.9643, 0.9648,
#         0.9648, 0.9654, 0.9653, 0.9660, 0.9670, 0.9676, 0.9673, 0.9674, 0.9692,
#         0.9694, 0.9699, 0.9707, 0.9711, 0.9713, 0.9718, 0.9726, 0.9729, 0.9736,
#         0.9736, 0.9747, 0.9754, 0.9757, 0.9757, 0.9774, 0.9773, 0.9784, 0.9783,
#         0.9793, 0.9795, 0.9803, 0.9806, 0.9818, 0.9824, 0.9839, 0.9843, 0.9857,
#         0.9858, 0.9868, 0.9867, 0.9874, 0.9871, 0.9875, 0.9851, 0.9853, 0.9784,
#         0.9772, 0.9777, 0.9766, 0.9765, 0.9764, 0.9767, 0.9766])

# tensor([0.7456, 0.9496, 0.9603, 0.9614, 0.9629, 0.9627, 0.9636, 0.9635, 0.9637,
#         0.9636, 0.9642, 0.9643, 0.9653, 0.9658, 0.9663, 0.9664, 0.9671, 0.9687,
#         0.9691, 0.9698, 0.9703, 0.9708, 0.9710, 0.9708, 0.9719, 0.9725, 0.9733,
#         0.9733, 0.9739, 0.9745, 0.9751, 0.9755, 0.9764, 0.9763, 0.9768, 0.9781,
#         0.9791, 0.9792, 0.9806, 0.9813, 0.9824, 0.9830, 0.9836, 0.9841, 0.9852,
#         0.9853, 0.9861, 0.9864, 0.9875, 0.9870, 0.9878, 0.9833, 0.9843, 0.9827,
#         0.9819, 0.9798, 0.9789, 0.9781, 0.9786, 0.9781, 0.9790])

# legal_7M accs
# tensor([0.7110, 0.8477, 0.8864, 0.8873, 0.8881, 0.8895, 0.8895, 0.8952, 0.8953,
#         0.8970, 0.8971, 0.8981, 0.8982, 0.9044, 0.9046, 0.9052, 0.9052, 0.9115,
#         0.9120, 0.9141, 0.9143, 0.9152, 0.9152, 0.9159, 0.9164, 0.9224, 0.9228,
#         0.9268, 0.9271, 0.9323, 0.9329, 0.9334, 0.9338, 0.9358, 0.9364, 0.9393,
#         0.9395, 0.9435, 0.9439, 0.9473, 0.9482, 0.9521, 0.9540, 0.9558, 0.9577,
#         0.9587, 0.9611, 0.9619, 0.9666, 0.9672, 0.9738, 0.9754, 0.9875, 0.9880,
#         0.9961, 0.9961, 0.9967, 0.9967, 0.9966, 0.9966, 0.9962])

# tem_caps
# tensor([0.0411, 0.8573, 0.9569, 0.9577, 0.9624, 0.9627, 0.9629, 0.9632, 0.9635,
#         0.9639, 0.9637, 0.9643, 0.9654, 0.9656, 0.9662, 0.9662, 0.9672, 0.9688,
#         0.9690, 0.9692, 0.9701, 0.9700, 0.9707, 0.9708, 0.9713, 0.9723, 0.9734,
#         0.9733, 0.9740, 0.9744, 0.9750, 0.9754, 0.9761, 0.9765, 0.9769, 0.9777,
#         0.9790, 0.9794, 0.9806, 0.9812, 0.9824, 0.9830, 0.9837, 0.9841, 0.9850,
#         0.9851, 0.9862, 0.9865, 0.9870, 0.9870, 0.9878, 0.9833, 0.9844, 0.9828,
#         0.9823, 0.9795, 0.9780, 0.9784, 0.9782, 0.9784, 0.9785])

# tensor([0.0648, 0.8639, 0.9533, 0.9549, 0.9604, 0.9607, 0.9611, 0.9614, 0.9614,
#         0.9612, 0.9621, 0.9625, 0.9632, 0.9635, 0.9643, 0.9644, 0.9653, 0.9663,
#         0.9665, 0.9671, 0.9677, 0.9676, 0.9684, 0.9684, 0.9694, 0.9696, 0.9707,
#         0.9710, 0.9719, 0.9719, 0.9725, 0.9731, 0.9742, 0.9742, 0.9747, 0.9762,
#         0.9773, 0.9777, 0.9785, 0.9795, 0.9805, 0.9812, 0.9820, 0.9828, 0.9835,
#         0.9837, 0.9848, 0.9849, 0.9858, 0.9855, 0.9863, 0.9811, 0.9820, 0.9805,
#         0.9782, 0.9759, 0.9746, 0.9739, 0.9740, 0.9742, 0.9746])

# ptem_7M
# tensor([0.3346, 0.6783, 0.7305, 0.7584, 0.7647, 0.7717, 0.7743, 0.7871, 0.7893,
#         0.7958, 0.7972, 0.8082, 0.8102, 0.8204, 0.8217, 0.8303, 0.8306, 0.8640,
#         0.8644, 0.8731, 0.8727, 0.8786, 0.8787, 0.8828, 0.8828, 0.8948, 0.8945,
#         0.8984, 0.8977, 0.9056, 0.9047, 0.9160, 0.9150, 0.9226, 0.9215, 0.9318,
#         0.9303, 0.9403, 0.9384, 0.9480, 0.9464, 0.9523, 0.9502, 0.9556, 0.9529,
#         0.9595, 0.9574, 0.9658, 0.9637, 0.9708, 0.9688, 0.9732, 0.9685, 0.9702,
#         0.9659, 0.9668, 0.9610, 0.9598, 0.9589, 0.9583, 0.9581])

# %%
probes = load_probes(
    probe_dir, device, w_u=model.W_U.detach(), w_e=model.W_E.T.detach(), combos=["t+m", "t-m", "t-e"]
)
{k: p.shape for k, p in probes.items()}  # d_model row col n_probe_layer

# %%
linear_probe_tem = t.stack([probes[k] for k in "tem"], dim=-2)
linear_probe_cap = t.stack([-probes["c"], probes["c"]], dim=-2)
linear_probe_legal = t.stack([-probes["l"], probes["l"]], dim=-2)
# linear_probe_tem_caps = t.stack([probes[k+"c"] for k in "tnm"], dim=-2)
linear_probe_ptem = t.stack([probes["p"+k] for k in "tem"], dim=-2)

# %%
batch = dataset_dict["test"].take(1)
plot_game(batch[0])
probe_layer = 54
plot_probe_preds(
    model,
    device,
    linear_probe_legal,
    batch,
    target_fn=lambda x: legality_target(x, device),
    layer=probe_layer,
    index=0,
    title="Legal probe",
)
plot_probe_preds(
    model,
    device,
    linear_probe_tem,
    batch,
    target_fn=lambda x: theirs_empty_mine_target(x, device),
    layer=probe_layer,
    index=0,
    title="TEM probe",
)
plot_probe_preds(
    model,
    device,
    linear_probe_ptem,
    batch,
    target_fn=lambda x: prev_tem_target(x, device),
    layer=probe_layer,
    index=0,
    title="PTEM probe",
)
plot_probe_preds(
    model,
    device,
    linear_probe_cap,
    batch,
    target_fn=lambda x: captures_target(x, device),
    layer=probe_layer,
    index=0,
    title="Captures probe",
)

# %%
# Visualise cross-orthogonality between linear probes, across layers and features
dot_names, dot_probes = zip(*probes.items())
dot_probes = t.stack(tuple(dot_probes), dim=-1)
probe_layers = [0, 50, 60]
dot_probes = dot_probes[..., probe_layers, :]
dots = einops.einsum(
    dot_probes,
    dot_probes,
    "d_model row col layer_0 probe_0, d_model row col layer_1 probe_1 -> row col probe_0 probe_1 layer_0 layer_1",
)
# probe_layers = list(range(dot_probes.shape[-2]))
dots = einops.reduce(
    dots,
    "row col probe_0 probe_1 layer_0 layer_1 -> (layer_0 probe_0) (layer_1 probe_1)",
    t.nanmean
)
dots = t.tril(dots)

index = [
    f"L{l} {probe_name}"
    for l, probe_name in product(probe_layers, dot_names)
]
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
all_probes = [
    linear_probe_tem,
    linear_probe_cap,
    linear_probe_legal,
]
all_probes = t.cat([p[:, :, :, :, 1:-1] for p in all_probes], dim=-2)
all_probes /= all_probes.norm(dim=0, keepdim=True)
positional_dots = einops.einsum(
    all_probes[:, : size // 2, : size // 2],
    all_probes,
    "d_model r0 c0 n_probe n_layer, d_model r1 c1 n_probe n_layer -> r0 c0 r1 c1 n_probe n_layer",
)
# positional_dots[*(range(size) for _ in range(4))] = 0
positional_dots = einops.reduce(
    positional_dots, "r0 c0 r1 c1 n_probe n_layer -> (r0 c0 n_probe) r1 c1", "mean"
).cpu()
positional_names = ["t", "e", "m", "c", "nc", "l", "nl"]
plot_game(
    {"boards": positional_dots},
    hovertext=positional_dots,
    reversed=False,
    subplot_titles=[
        f"{chr(ord('A') + x)}{y + 1} {p}"
        for y in range(size // 2)
        for x in range(size // 2)
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
