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
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots

from othello_gpt.data.vis import plot_game
from othello_gpt.model.nanoGPT import GPT, GPTConfig
from othello_gpt.util import (
    convert_nanogpt_to_transformer_lens_weights,
    get_all_squares,
    vocab_to_board,
)
from othello_gpt.research.targets import captures_target, forward_probe

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
    vocab_size=size * size - 4,  # no pad
    n_layer=2,
    n_head=4,
    n_embd=256,
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
probe_names = {
    "tem": "linear_probe_20250217_173756_tem_256.pt",
    "l": "linear_probe_20250217_191136_legal_256.pt",
    "c": "linear_probe_20250217_182206_cap_256.pt",
    ("pt", "pe", "pm"): "linear_probe_20250219_003309_ptem_256.pt",
}
probes = {}
for names, file in probe_names.items():
    probe = t.load(probe_dir / file, weights_only=True, map_location=device).detach()
    probe /= probe.norm(dim=0, keepdim=True)
    for i, n in enumerate(names):
        probes[n] = probe[..., i, :]
{k: p.shape for k, p in probes.items()}

# # %%
# focus_games = dataset_dict["test"].take(10)
# focus_input_ids = t.tensor(focus_games["input_ids"], device=device)
# focus_logits = model(focus_input_ids[:, :-1]).log_softmax(dim=-1)
# focus_prob_boards = vocab_to_board(t.exp(focus_logits.detach().cpu()), size)
# focus_flip_boards = t.tensor()

# flips_pred_logprob, _ = forward_probe(
#     model,
#     device,
#     t.stack([probes["f"], probes["n"]], dim=3),
#     focus_games,
#     captures_target,
#     return_loss=False,
# )
# flips_pred_prob = t.exp(flips_pred_logprob).to("cpu")
# flips_pred_prob, flips_pred_index = flips_pred_prob.max(dim=-1)

# # %%
# probe_layer = 2
# test_index = 0
# test_game = focus_games[test_index]
# plot_game(test_game)
# test_flips = t.tensor(test_game["flips"], dtype=int)
# plot_game(
#     {"boards": test_flips, "moves": test_game["moves"]},
#     reversed=False,
#     textcolor="white",
# )
# test_pred_model = {
#     "boards": focus_prob_boards[test_index],
#     "legalities": test_game["legalities"],
#     "moves": test_game["moves"],
# }
# flips_dict = {
#     "boards": flips_pred_index[probe_layer, test_index],
#     "legalities": focus_games["flips"][test_index],
#     "moves": test_game["moves"],
# }
# plot_game(
#     flips_dict,
#     reversed=False,
#     textcolor="red",
#     hovertext=flips_pred_prob[probe_layer, test_index],
#     shift_legalities=False,
# )
# plot_game(
#     test_pred_model,
#     reversed=False,
#     textcolor="red",
#     hovertext=test_pred_model["boards"],
#     title="Model predictions for legal moves",
# )


# # %%
# def apply_scale(
#     resid: Float[t.Tensor, "batch seq d_model"],
#     flip_dir: Float[t.Tensor, "d_model"],
#     scale: int,
#     pos: int,
# ) -> Float[t.Tensor, "batch seq d_model"]:
#     """
#     Returns a version of the residual stream, modified by the amount `scale` in the
#     direction `flip_dir` at the sequence position `pos`, in the way described above.
#     """
#     v = flip_dir / flip_dir.norm(keepdim=True)
#     a = einops.einsum(resid[:, pos, :], v, "batch d_model, d_model -> batch")
#     new_resid = resid.detach().clone()
#     new_resid[:, pos, :] -= a * (1 + scale) * v
#     return new_resid

# # %%
# layer = 1
# move = 10
# scale = 2
# flip_coords = (2, 4)

# def flip_hook(resid: Float[t.Tensor, "batch seq d_model"], hook: HookPoint):
#     return apply_scale(resid, probes["f"][:, *flip_coords, 2 * layer], scale, move)

# hooked_logits = model.run_with_hooks(
#     test_input_ids[:, :-1],
#     fwd_hooks=[
#         (get_act_name("resid_pre", layer), flip_hook)
#     ],
# ).log_softmax(dim=-1)[0, move]

# hooked_prob_board = vocab_to_board(t.exp(hooked_logits.detach().cpu()), size)

# %%
fig = make_subplots(1, 2)
fig.add_trace(
    go.Heatmap(
        z=prob_boards[move],
        colorscale="gray",
        showscale=False,
    ), row=1, col=1,
)
fig.add_trace(
    go.Heatmap(
        z=hooked_prob_board,
        colorscale="gray",
        showscale=False,
    ), row=1, col=2,
)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    linewidth=1,
    mirror=True,
    constrain="domain",
    autorange="reversed",
)
fig.update_xaxes(
    showline=True,
    linecolor="black",
    linewidth=1,
    mirror=True,
    scaleanchor="y",
    scaleratio=1,
    constrain="domain",
)
subplot_size = 180
margin = subplot_size // 8
fig.update_layout(
    font=dict(size=subplot_size // 20),
    margin=dict(l=margin, r=margin, t=margin * 3, b=margin),
    width=subplot_size * 2,
    height=subplot_size,
)
fig.show()

# %%
