# %%
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union, List, Tuple
from itertools import product
import pandas as pd

import einops
import huggingface_hub as hf
import numpy as np
import torch as t
import wandb
from datasets import load_dataset
from eindex import eindex
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
import plotly.graph_objects as go

from othello_gpt.data.vis import plot_game
from othello_gpt.model.nanoGPT import GPT, GPTConfig
from othello_gpt.util import (
    convert_nanogpt_to_transformer_lens_weights,
    get_all_squares,
)

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
class HubGPT(GPT, hf.PyTorchModelHubMixin):
    pass


nano_cfg = GPTConfig(
    # block_size=(size * size - 4) * 2 - 1,
    block_size=(size * size - 4) - 1,
    # vocab_size=size * size - 4 + 2,  # pass and pad
    vocab_size=size * size - 4 + 1,  # pad
    n_layer=2,
    n_head=2,
    n_embd=128,
    dropout=0.0,
    bias=True,
)
hooked_cfg = HookedTransformerConfig(
    n_layers=nano_cfg.n_layer,
    d_model=nano_cfg.n_embd,
    n_ctx=nano_cfg.block_size,
    d_head=nano_cfg.n_embd // nano_cfg.n_head,
    n_heads=nano_cfg.n_head,
    d_vocab=nano_cfg.vocab_size,
    act_fn="gelu",
    normalization_type="LN",
    device=device,
)

model = HubGPT.from_pretrained("awonga/othello-gpt", config=nano_cfg).to(device)
state_dict = convert_nanogpt_to_transformer_lens_weights(
    model.state_dict(), nano_cfg, hooked_cfg
)
model = HookedTransformer(hooked_cfg)
model.load_and_process_state_dict(state_dict)
model.to(device)

# %%
def theirs_empty_mine_target(batch):
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    boards[:, 1::2] *= -1
    boards += 1
    return boards


def legality_target(batch):
    legal = t.tensor(batch["legalities"], device=device)[:, 1:]
    return legal


def flip_parity_target(batch):
    # At each game position, a non-empty tile is either the same colour as when it was first played (0)
    # or it has been flipped to the other colour (1). I think that this is a necessary state for the
    # model to track in order to have an accurate board state representation.
    flips = t.tensor(batch["flips"], device=device)[:, :-1].int()
    return flips.cumsum(dim=1) % 2


def mine_flip_target(batch):
    # TODO (why) is initial accuracy on last layer significantly higher?
    # Hypothesis: model tracks tiles that got flipped, grouped by mine/theirs
    coords = t.tensor(batch["coords"], device=device)
    boards = t.tensor(batch["boards"], device=device)[:, :-1]

    original_colour_boards = t.zeros((boards.shape[0], size, size))
    original_colour_boards[:, [2, 3], [2, 3]] = 1
    original_colour_boards[:, [2, 3], [3, 2]] = -1
    for i, (ys, xs) in list(enumerate(zip(coords[:, :, 0], coords[:, :, 1]))):
        original_colour_boards[i, ys[::2], xs[::2]] = 1
        original_colour_boards[i, ys[1::2], xs[1::2]] = -1
    original_colour_boards = einops.repeat(
        original_colour_boards,
        "n_batch row col -> n_batch pos row col",
        pos=boards.shape[1],
    )
    flip_boards = (boards != original_colour_boards) & (boards != 0)

    # Either: flipped and currently mine = 2, flipped and currently theirs = 0, else 1
    # target_boards = t.where(flip_boards, boards, 0)
    # Or: flipped and originally mine
    target_boards = t.where(flip_boards, original_colour_boards, 0)

    target_boards[:, 1::2] *= -1
    target_boards += 1
    return target_boards


def captures_target(batch):
    # Hypothesis: each token tracks the tiles that it captured when the move was played
    # After H0, we have [my moves; their moves; my moves flipped; their moves flipped]
    # This gives us the
    return t.tensor(batch["flips"], device=device)[:, :-1].int()


def original_colour_target(batch):
    # # Hypothesis: moves -> H0 -> (original colour, flips) -> M0 -> (originally mine & not flipped = mine, originally mine & flipped = theirs, empty = empty, etc.) -> H1 -> (?) -> M1 -> (legal)
    return t.tensor(batch["originals"], device=device)[:, :-1].int() + 1


def forward_probe(
    linear_probe, batch, target_fn, return_loss=True, return_labels=False
) -> Tuple[t.Tensor, Union[t.Tensor, List]]:
    # input_ids = pad_batch(batch["input_ids"], max_len=model.cfg.n_ctx + 1).to(device)
    input_ids = t.tensor(batch["input_ids"], device=device)
    _, cache = model.run_with_cache(
        input_ids[:, :-1],
        names_filter=lambda name: "hook_resid_" in name
        or "ln_final.hook_scale" in name,
    )
    X, labels = cache.accumulated_resid(
        apply_ln=True, incl_mid=True, return_labels=True
    )

    preds = einops.einsum(
        X,
        linear_probe,
        "layer batch n_ctx d_model, d_model row col d_probe layer -> layer batch n_ctx row col d_probe",
    )
    log_probs = preds.log_softmax(-1)

    if not return_loss:
        if return_labels:
            return log_probs, labels
        return log_probs, None

    y = target_fn(batch)
    correct_log_probs = eindex(
        log_probs, y, "layer batch n_ctx rows cols [batch n_ctx rows cols]"
    )
    loss = -correct_log_probs.mean()

    return log_probs, loss


target_fn = original_colour_target

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
    n_epochs: int = 12
    lr: float = 1e-3
    batch_size: int = 1024
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
        test_y_pred, test_loss = forward_probe(linear_probe, test_dataset, target_fn)
    test_accs = (test_y_pred > np.log(0.5)).argmax(-1) == test_y
    test_accs = einops.reduce(
        test_accs.float(), "layer batch pos row col -> layer", "mean"
    )
    test_accs = test_accs.cpu().round(decimals=4)
    return test_loss, test_accs


def train_linear_probe(
    model: HookedTransformer,
    args: LinearProbeTrainingArgs,
    target_fn: Callable,
):
    test_dataset = dataset_dict["test"].take(args.n_test)
    test_y = target_fn(test_dataset).to(device)
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
            _, loss = forward_probe(linear_probe, batch, target_fn)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.update()
            pbar.set_description(
                f"Epoch {i + 1}/{args.n_epochs} {loss=:.4f} {test_accs=}"
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


args = LinearProbeTrainingArgs()
# args = LinearProbeTrainingArgs(
#     use_wandb=False, n_epochs=2, n_steps_per_epoch=10, lr=1e-3
# )
# linear_probe = train_linear_probe(model, args, target_fn)

# %%
# t.save(
#     linear_probe,
#     probe_dir / f"linear_probe_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_mid_otem.pt",
# )
linear_probe_otem = t.load(
    probe_dir / "linear_probe_20250213_221416_mid_otem.pt",
    weights_only=True,
    map_location=device,
).detach()
linear_probe_tem = t.load(
    probe_dir / "linear_probe_20250212_141120_mid_tem.pt",
    weights_only=True,
    map_location=device,
).detach()
linear_probe_flips = t.load(
    probe_dir / "linear_probe_20250213_000355_mid_flips.pt",
    weights_only=True,
    map_location=device,
).detach()
linear_probe_otem.shape, linear_probe_tem.shape, linear_probe_flips.shape  # d_model row col d_probe n_layer

# %% 
# TODO ex: why are the sums of W_U approx 0?

# %%
# Visualise cross-orthogonality between linear probes, across layers and features
# Visualise additivity between exhaustive probe spaces (e.g. theirs + empty + mine = 1, flipped + not flipped = 1)
otheirs = linear_probe_otem[:, :, :, 0].flatten(1, -2)[:, all_squares]
oempty = linear_probe_otem[:, :, :, 1].flatten(1, -2)[:, all_squares]
omine = linear_probe_otem[:, :, :, 2].flatten(1, -2)[:, all_squares]
theirs = linear_probe_tem[:, :, :, 0].flatten(1, -2)[:, all_squares]
empty = linear_probe_tem[:, :, :, 1].flatten(1, -2)[:, all_squares]
mine = linear_probe_tem[:, :, :, 2].flatten(1, -2)[:, all_squares]
flipped = linear_probe_flips[:, :, :, 0].flatten(1, -2)[:, all_squares]
not_flipped = linear_probe_flips[:, :, :, 1].flatten(1, -2)[:, all_squares]
r0 = t.randn_like(theirs)
r1 = t.randn_like(r0)
unembed = einops.repeat(model.W_U[:, 1:], "d_model n_vocab -> d_model n_vocab n_layer", n_layer=otheirs.shape[-1]).detach().cpu()
names = [
    "otheirs",
    # "oempty",
    "omine",
    "otheirs+omine",
    "theirs",
    "empty",
    "mine",
    "theirs+mine",
    "flipped",
    # "not_flipped",
    "omine+flipped",
    # "omine+not_flipped",
    # "r0",
    # "r1",
    "unembed",
] # TODO add W_U
layers = [f"L{i}" for i in range(linear_probe_tem.shape[-1])]
probes = t.stack([
    otheirs,
    # oempty,
    omine,
    otheirs + omine,
    theirs,
    empty,
    mine,
    theirs + mine,
    flipped,
    # not_flipped,
    omine + flipped,
    # omine + not_flipped,
    # r0,
    # r1,
    unembed,
])
probes /= probes.norm(dim=1, keepdim=True)
dots = einops.einsum(probes, probes, "probe_0 d_model n layer_0, probe_1 d_model n layer_1 -> n probe_0 probe_1 layer_0 layer_1")
dots = einops.reduce(dots, "n probe_0 probe_1 layer_0 layer_1 -> (layer_0 probe_0) (layer_1 probe_1)", "mean")
dots = t.tril(dots)
index = [f"{layer_name} {probe_name}" for layer_name, probe_name in product(layers, names)]
dots_df = pd.DataFrame(dots.cpu(), index=index, columns=index)

fig = go.Figure(
    data=go.Heatmap(
        z=dots_df.values,
        x=dots_df.columns,
        y=dots_df.index,
        colorscale='RdBu',
        zmid=0,
    ),
)

fig.update_layout(
    title="Heatmap of Dot Products Between Probes",
    height=720,
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(index))),
        ticktext=dots_df.columns,
        tickangle=30,
        tickfont=dict(size=8),
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(index))),
        ticktext=dots_df.index,
        tickfont=dict(size=8),
    ),
    shapes=[
        dict(
            type="line",
            x0=i * len(names) - 0.5,
            y0=-0.5,
            x1=i * len(names) - 0.5,
            y1=len(index) - 0.5,
            line=dict(color="black", width=1),
        )
        for i in range(1, len(layers))
    ] + [
        dict(
            type="line",
            x0=-0.5,
            y0=i * len(names) - 0.5,
            x1=len(index) - 0.5,
            y1=i * len(names) - 0.5,
            line=dict(color="black", width=1),
        )
        for i in range(1, len(layers))
    ],
)

fig.show()

# %%
# Visualise orthogonality between feature probes for different squares
all_probes = [linear_probe_otem, linear_probe_tem, linear_probe_flips]
all_probes = t.cat([p[:, :, :, :, 1:-1] for p in all_probes], dim=-2)
all_probes /= all_probes.norm(dim=0, keepdim=True)
positional_dots = einops.einsum(all_probes[:, :size//2, :size//2], all_probes, "d_model r0 c0 n_probe n_layer, d_model r1 c1 n_probe n_layer -> r0 c0 r1 c1 n_probe n_layer")
# positional_dots[*(range(size) for _ in range(4))] = 0
positional_dots = einops.reduce(positional_dots, "r0 c0 r1 c1 n_probe n_layer -> (r0 c0 n_probe) r1 c1", "mean")
plot_game(
    {"boards": positional_dots},
    hovertext=positional_dots,
    reversed=False,
    subplot_titles=[f"{chr(ord('A') + x)}{y + 1} {p}" for y in range(size//2) for x in range(size//2) for p in ["ot", "oe", "om", "t", "e", "m", "f", "nf"]],
    shift_legalities=False,
    n_cols=12,
)

# %%
n_focus = 1000
focus_games = dataset_dict["test"].take(n_focus)
focus_input_ids = t.tensor(focus_games["input_ids"], device=device)
# focus_logprobs = model(focus_input_ids[:, :-1])
# focus_prob_boards = t.zeros((n_focus, size, size))
# focus_logits, focus_cache = model.run_with_cache(focus_input_ids[:, :-1])
focus_logits = model(focus_input_ids[:, :-1])
focus_logit_boards = t.full((n_focus, focus_logits.shape[1], size, size), 0.0)
focus_logit_boards.flatten(2)[..., all_squares] = focus_logits[..., 1:].detach().cpu()
focus_probs = focus_logits.softmax(-1)
focus_prob_boards = t.full((n_focus, focus_logits.shape[1], size, size), 0.0)
focus_prob_boards.flatten(2)[..., all_squares] = focus_probs[..., 1:].detach().cpu()
focus_otem_boards = original_colour_target(focus_games).cpu()
focus_tem_boards = theirs_empty_mine_target(focus_games).cpu()
focus_flip_boards = captures_target(focus_games).cpu()
focus_flip_parity = flip_parity_target(focus_games).cpu()

# X, labels = focus_cache.accumulated_resid(incl_mid=True, return_labels=True)
# y = target_fn(focus_games)
with t.no_grad():
    otem_pred_logprob, labels = forward_probe(
        linear_probe_otem,
        focus_games,
        original_colour_target,
        return_loss=False,
        return_labels=True,
    )
    tem_pred_logprob, _ = forward_probe(
        linear_probe_tem,
        focus_games,
        theirs_empty_mine_target,
        return_loss=False,
    )
    flips_pred_logprob, _ = forward_probe(
        linear_probe_flips, focus_games, captures_target, return_loss=False
    )
    otem_pred_prob = t.exp(otem_pred_logprob).to("cpu")
    tem_pred_prob = t.exp(tem_pred_logprob).to("cpu")
    flips_pred_prob = t.exp(flips_pred_logprob).to("cpu")
    otem_pred_prob, otem_pred_index = otem_pred_prob.max(dim=-1)
    tem_pred_prob, tem_pred_index = tem_pred_prob.max(dim=-1)
    flips_pred_prob, flips_pred_index = flips_pred_prob.max(dim=-1)

# %%
test_index = 0
n_cols = 8
subplot_size = 120
test_moves = focus_games[test_index]["moves"]

test_pred_model = {
    "boards": focus_prob_boards[test_index].detach().cpu(),
    "legalities": focus_games[test_index]["legalities"],
    "moves": test_moves,
}

for layer, name in enumerate(labels):
    otem_dict = {
        "boards": otem_pred_index[layer, test_index],
        "legalities": focus_otem_boards[test_index] == 1,
        "moves": test_moves,
    }
    tem_dict = {
        "boards": tem_pred_index[layer, test_index],
        "legalities": focus_tem_boards[test_index] == 1,
        "moves": test_moves,
    }
    flips_dict = {
        "boards": flips_pred_index[layer, test_index],
        "legalities": focus_flip_boards[test_index],
        "moves": test_moves,
    }

    plot_game(
        focus_games[test_index],
        title="Ground truth board states and legal moves",
        n_cols=n_cols,
        subplot_size=subplot_size,
    )
    plot_game(
        test_pred_model,
        reversed=False,
        textcolor="red",
        hovertext=test_pred_model["boards"],
        title="Model predictions for legal moves",
        n_cols=n_cols,
        subplot_size=subplot_size,
    )
    plot_game(
        otem_dict,
        reversed=False,
        textcolor="red",
        hovertext=otem_pred_prob[layer, test_index],
        shift_legalities=False,
        title=f"Layer {name} linear probe prediction for original colours",
        n_cols=n_cols,
        subplot_size=subplot_size,
    )
    plot_game(
        tem_dict,
        reversed=False,
        textcolor="red",
        hovertext=tem_pred_prob[layer, test_index],
        shift_legalities=False,
        title=f"Layer {name} linear probe prediction for board state",
        n_cols=n_cols,
        subplot_size=subplot_size,
    )
    plot_game(
        flips_dict,
        reversed=False,
        textcolor="red",
        hovertext=flips_pred_prob[layer, test_index],
        shift_legalities=False,
        title=f"Layer {name} linear probe prediction for flipped tiles",
        n_cols=n_cols,
        subplot_size=subplot_size,
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
# See how n flips affects accuracy
focus_n_flips = focus_flip_boards.flatten(-2, -1).sum(dim=-1)

focus_coords_pred = focus_logit_boards.view(
    focus_logit_boards.shape[0], focus_logit_boards.shape[1], -1
).argmax(dim=-1)
focus_coords_pred = t.stack(
    (focus_coords_pred // size, focus_coords_pred % size), dim=-1
)
focus_legalities = t.tensor(focus_games["legalities"])[:, 1:]

focus_pred_is_legal = t.zeros((n_focus, focus_coords_pred.shape[1]), dtype=t.bool)
for i in range(n_focus):
    for j in range(focus_coords_pred.shape[1]):
        y, x = focus_coords_pred[i, j]
        focus_pred_is_legal[i, j] = focus_legalities[i, j, y, x]

cond_accs = t.tensor([
    (
        focus_n_flips[:, i][focus_pred_is_legal[:,i]].float().mean(),
        focus_n_flips[:, i][~focus_pred_is_legal[:, i]].float().mean(),
    )
    for i in range(model.cfg.n_ctx)
])
fig = go.Figure()

fig.add_trace(go.Scatter(y=cond_accs[:, 0], mode='lines', name='Mean Flips (Legal)'))
fig.add_trace(go.Scatter(y=cond_accs[:, 1], mode='lines', name='Mean Flips (Illegal)'))

fig.update_layout(
    title="Mean Number of Flips for Legal and Illegal Moves",
    xaxis=dict(title="Move #"),
    yaxis=dict(title="Mean Number of Flips"),
)

fig.show()

# %%
# Find out how H0 constructs an 89% accurate board state
# Hypothesis: each token tracks the tiles that it captured when the move was played
# After H0, we have [my moves; their moves; my moves flipped; their moves flipped]
# Linear probe can then +- to get the board state
