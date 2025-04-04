# %%
from itertools import product
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from datasets import load_dataset
from jaxtyping import Float
from plotly.subplots import make_subplots
from torch.types import Tensor

from othello_gpt.data.vis import move_id_to_text, plot_game
from othello_gpt.model.sae import SAE, SAEConfig
from othello_gpt.util import get_all_squares, load_model, load_probes

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

root_dir = Path().cwd()  # .parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"

model_version = "600k"
model_name = f"awonga/othello-gpt-{model_version}"
model = load_model(device, model_name)
model.requires_grad_(False)

dataset_dict = load_dataset("awonga/othello-gpt")
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]

size = 6
all_squares = get_all_squares(size)
actually_all_squares = list(range(size * size))
corners = [0, size - 1, size * (size - 1), size * size - 1]
non_corners = [i for i in range(size * size) if i not in corners]
edges = [
    y * size + x
    for y in range(size)
    for x in range(size)
    if x * y == 0 or x == size - 1 or y == size - 1
]
non_edges = [i for i in range(size * size) if i not in edges]

# %%
attn_hook_suffixes = ("attn.hook_z", "attn.hook_z")
mlp_hook_suffixes = ("ln2.hook_normalized", "hook_mlp_out")
hook_suffixes = [attn_hook_suffixes, mlp_hook_suffixes]
cfgs = [
    SAEConfig(
        d_in=model.cfg.d_model,
        d_sae=1024,
        in_hook_layer=i,
        out_hook_layer=i,
        in_hook_suffix=in_hook_suffix,
        out_hook_suffix=out_hook_suffix,
    )
    for i, (in_hook_suffix, out_hook_suffix) in product(
        range(model.cfg.n_layers), hook_suffixes
    )
]
hook_names = [
    f"blocks.{cfg.out_hook_layer}.{cfg.out_hook_suffix}"
    if cfg.in_hook_suffix == cfg.out_hook_suffix
    else f"blocks.{cfg.in_hook_layer}.{cfg.in_hook_suffix}-blocks.{cfg.out_hook_layer}.{cfg.out_hook_suffix}"
    for cfg in cfgs
]
sae_names = [f"{model_name}-sae-{hook_name}" for hook_name in hook_names]
saes = [
    SAE.from_pretrained(
        sae_name,
        cfg=cfg,
        model=model,
        device=device,
    )
    for sae_name, cfg in zip(sae_names, cfgs)
]
for sae in saes:
    sae.eval()
    sae.requires_grad_(False)

n_test = 1024
batched_test_dataset = test_dataset.take(n_test).batch(128)
with t.inference_mode():
    eval_dicts, test_forward_dicts = zip(
        *[sae.evaluate(batched_test_dataset) for sae in saes]
    )
    eval_dict = {k: [d[k] for d in eval_dicts] for k in eval_dicts[0]}
    test_forward_dict = {
        k: t.stack(
            [d[k].reshape(n_test, model.cfg.n_ctx, -1) for d in test_forward_dicts],
            dim=2,
        )
        for k in test_forward_dicts[0]
    }
eval_dict

# %%
padded_W_pos = t.full((size * size, model.W_pos.shape[1]), t.nan, device=device)
padded_W_pos[: model.W_pos.shape[0], :] = model.W_pos
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    w_p=padded_W_pos.T.detach(),
    combos=["+t-m"],
    model_version=model_version,
)
probes["pos"] = t.nn.functional.pad(
    probes["pos"], (0, 0, 0, size * size - probes["pos"].shape[1]), value=float("nan")
)
probe_keys = {
    "ee": all_squares,
    "+t-m": actually_all_squares,
    "c": non_corners,
    "mov": all_squares,
    "b": all_squares,
    "u": all_squares,
    "p": range(model.cfg.n_ctx),
    "l": all_squares,
    "pos": range(4),
}
probe_suffixes = {
    k: [f"{i if k in ['p', 'pr'] else move_id_to_text(i, size)}" for i in s]
    for k, s in probe_keys.items()
}
probe_labels = [f"{k}_{s}" for k in probe_keys for s in probe_suffixes[k]]
probes_normed: Float[Tensor, "d_model n_probe n_sae"] = t.cat(
    [probes[k][:, s, 1:] for k, s in probe_keys.items()], dim=1
)

# %%
latents = [model.W_E_pos / model.W_E_pos.norm(dim=-1, keepdim=True)]
latent_labels = [f"b_{move_id_to_text(i, size)}" for i in all_squares]
latent_labels += [f"p_{i}" for i in range(model.cfg.n_ctx)]

for sae in saes:
    layer = sae.cfg.out_hook_layer

    is_attn_z = (sae.in_hook_name == sae.out_hook_name) and (
        sae.in_hook_name.endswith("attn.hook_z")
    )
    is_transcoder = sae.in_hook_name.endswith(
        "ln2.hook_normalized"
    ) and sae.out_hook_name.endswith("hook_mlp_out")

    if is_attn_z:
        attn_latents = sae.W_dec_normalized.reshape(
            sae.cfg.d_sae, sae.model.cfg.n_heads, sae.model.cfg.d_head
        )
        w_o = sae.model.W_O[layer]
        head_latents = einops.einsum(
            attn_latents,
            w_o,
            "d_sae n_head d_head, n_head d_head d_model -> d_sae n_head d_model",
        )

        # Norm such that the original z vectors' out projections have norm 1
        # i.e. head_latents.sum(1).norm(dim=-1) should be equal to 1 everywhere
        head_latents /= head_latents.sum(1, keepdim=True).norm(dim=-1, keepdim=True)

        latents.append(head_latents.flatten(0, 1))
        latent_labels += [
            f"a{layer}l{i}h{j}" for i, j in product(*map(range, head_latents.shape[:2]))
        ]

    elif is_transcoder:
        latents.append(sae.W_dec_normalized)
        latent_labels += [f"m{layer}f{i}" for i in range(sae.cfg.d_sae)]

    else:
        raise ValueError("Unrecognised sae type")

latents.append((model.W_U / model.W_U.norm(dim=0, keepdim=True)).T)
latent_labels += [f"u_{move_id_to_text(i, size)}" for i in all_squares]

latent_idxs = t.tensor(
    np.cumsum([0] + [l.shape[0] for l in latents])[:-1], device=device
)
latents = t.cat(latents, dim=0)
len(latent_labels), latents.shape, latent_idxs

# %%
def node_to_latent_idx(node):
    return latent_idxs[node[0]] + node[1]


def node_to_label(node):
    latent_idx = node_to_latent_idx(node)
    return latent_labels[latent_idx]


def latent_idx_to_node(idx):
    layer = t.searchsorted(latent_idxs, idx, side="right") - 1
    offset = idx - latent_idxs[layer]
    return (layer.item(), offset.item())


k = 6  # expand by k times at each node
G = nx.Graph()

root = (len(latent_idxs) - 1, 19)
q = [(root, None, 0)]
while q:
    n, p, v = q.pop()
    G.add_node(n)
    if p is not None:
        G.add_edge(n, p, weight=v, abs_weight=abs(v))

    if n[0] == 0:  # embed layer => leaf node
        continue

    # A non-leaf node is either an unembed vector, a transcoder latent, or an attn_z latent
    latent_idx = node_to_latent_idx(n)
    latent_label = latent_labels[latent_idx]
    is_unembed = n[0] == len(latent_idxs) - 1
    is_transcoder = latent_label.startswith("m")
    is_attn_z = latent_label.startswith("a")

    # Unembed: find upstream latents that align with itself
    # Transcoder: find upstream latents that align with the in vector
    # Attn_z: find upstream latents projected in V space that align with itself

    if is_unembed:
        target_latent = latents[latent_idx]
    else:
        sae = saes[n[0] - 1]
        if is_transcoder:
            target_latent = sae.W_enc[:, n[1]]
            target_latent = target_latent / target_latent.norm(dim=0, keepdim=True)
        elif is_attn_z:
            # W_dec_normalized is (d_sae, d_in) -> (d_sae, n_head, d_head)
            # n[1] is in range(d_sae * n_head)
            d_head = sae.model.cfg.d_head
            target_latent = sae.W_dec_normalized.reshape(-1, d_head)[n[1]]
        else:
            raise ValueError("Unrecognised node", n)

    upstream_latents = latents[: latent_idxs[n[0]]]
    if is_attn_z:
        in_layer = sae.cfg.in_hook_layer
        n_head = sae.model.cfg.n_heads
        z_idx, h_idx = divmod(n[1], n_head)
        upstream_latents = upstream_latents @ sae.model.W_V[in_layer, h_idx]
    upstream_latents = upstream_latents / upstream_latents.norm(dim=-1, keepdim=True)

    upstream_alignments = upstream_latents @ target_latent
    if is_unembed:
        _, topk_latent_idxs = t.topk(upstream_alignments, k)  # stick to positive activations
    else:
        _, topk_latent_idxs = t.topk(upstream_alignments.abs(), k)

    for latent_idx in topk_latent_idxs:
        c = latent_idx_to_node(latent_idx)
        v = round(upstream_alignments[latent_idx].item(), 2)
        if abs(v) >= 0.5:
            q.append((c, n, v))

# Ensure all nodes have a subset_key attribute
for node in G.nodes:
    G.nodes[node]["subset_key"] = node[0]

# Generate the layout
layer_sizes = [latent_idxs[i + 1] - latent_idxs[i] for i in range(len(latent_idxs) - 1)] + [latents.shape[0] - latent_idxs[-1]]
pos = {}
for node in G.nodes:
    layer = node[0]
    offset = node[1]
    layer_size = layer_sizes[layer].item()
    pos[node] = (layer, 1 - offset / layer_size)
fig, ax = plt.subplots(figsize=(16, 32))  # Adjust the height by increasing the second value
nx.draw(
    G,
    pos,
    with_labels=True,
    labels={node: node_to_label(node) for node in G.nodes},
    node_size=800,
    font_size=8,
    edge_color=[G[u][v]["abs_weight"] for u, v in G.edges],
    edge_cmap=plt.cm.BuGn,
    edge_vmin=-1,
    edge_vmax=1,
    ax=ax,
)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    rotate=False,
    font_size=6,  # Make edge labels smaller
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="none"),  # Transparent background
)
plt.title(node_to_label(root))
plt.show()

# %%
game_idx = 0
game = test_dataset[game_idx]
sae_node_labels = []
acts = []
for n in sorted(G.nodes, key=lambda n: n[0]):
    if n[0] == 0 or n[0] == len(latent_idxs) - 1:
        continue

    sae_idx = n[0] - 1
    sae = saes[sae_idx]
    if sae.out_hook_name.endswith("attn.hook_z"):
        n_head = sae.model.cfg.n_heads
        z_idx, h_idx = divmod(n[1], n_head)
        acts.append(test_forward_dict["acts_pre"][game_idx, :, sae_idx, z_idx])
    else:
        acts.append(test_forward_dict["acts_pre"][game_idx, :, sae_idx, n[1]])
    sae_node_labels.append(node_to_label(n))

acts = t.stack(acts, dim=-1)
df = pd.DataFrame(acts.cpu(), columns=sae_node_labels)
def highlight_positive(val):
    return "font-weight: bold; color: red" if val > 0 else ""

df = df.loc[:, (df > 0).any()]  # Remove columns with no positive values
styled_df = df.style.format("{:.2f}").map(highlight_positive)
display(styled_df)
plot_game(game, subplot_size=120)

# %%
# Construct a priori auto-interps
# Weighted avg by acts_pre
#   1. Legalities
#   2. Position
#   3. Just moved
#   4. Just captured
#   5. Capture count
#   6. Capture parity
#   7. TEM
#   8. BEW
game.keys()

# %%
# Outputs:
#   - board_states tensor
#       shape: (latent, state, row, col)
#       desc: act-weighted average binary board state, e.g. l/t/e/m/b/w/c/mov/parity
#   - misc list
#       shape: [latent, ]

t.tensor(test_dataset.take(1000)["legalities"])[:, :-1].shape

