# %%
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import torch as t
import wandb
from datasets import load_dataset
from jaxtyping import Float
import numpy as np

from othello_gpt.data.vis import (
    plot_game,
    plot_probe_preds,
    plot_in_basis,
    move_id_to_text,
)
from othello_gpt.util import (
    get_all_squares,
    load_model,
    load_probes,
)

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"
probe_dir.mkdir(parents=True, exist_ok=True)

wandb.login()

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
dataset_dict = load_dataset("awonga/othello-gpt")
n_test = 1000
test_dataset = dataset_dict["test"].take(n_test)

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# %%
version = "6M"
model = load_model(device, f"awonga/othello-gpt-{version}")
model

# %%
padded_W_pos = t.full((size * size, model.W_pos.shape[1]), t.nan, device=device)
padded_W_pos[: model.W_pos.shape[0], :] = model.W_pos
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    w_p=padded_W_pos.T.detach(),
    combos=[
        "+t-m",
    ],
    model_version=version,
)

colors = dict(
    zip(
        [*probes.keys(), "tem"],
        px.colors.qualitative.Light24 + px.colors.qualitative.Dark24,
    )
)

{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
dict_probes = [
    # ("b", all_squares, 0),
    # ("p", range(model.cfg.n_ctx), 0),
    ("u", all_squares, 0),
    # ("ee", all_squares, 2),
    # ("+t-m", actually_all_squares, 13),
    # ("l", all_squares, 15),
    # ("c", non_corners, 12),
    # ("mov", all_squares, 2),
    # ("pos", range(3), 2),
]
dict_DN = t.cat([probes[k][:, s, l] for k, s, l in dict_probes], dim=1)
dict_labels = [
    f"{k}_{i}" if k in ["pos", "p"] else f"{k}_{move_id_to_text(i, size)}_L{l}"
    for k, s, l in dict_probes
    for i in s
]
dict_indices = np.cumsum([0] + [len(s) for _, s, _ in dict_probes])
dict_DN.shape, len(dict_labels), dict_indices

# %%
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp

embed_weights = [
    (model.W_E, "b", [f"b_{move_id_to_text(i, size)}" for i in all_squares]),
    (model.W_pos, "p", [f"p_{i}" for i in range(model.cfg.n_ctx)]),
]

qkv_weights = [
    (
        w_qkv[l, h].transpose(-2, -1),
        f"{qkv}_L{l}H{h}",
        [
            f"{qkv}_L{l}H{h}D{d}"
            for d in range(d_head)
        ],
    )
    for l in range(model.cfg.n_layers)
    for h in range(model.cfg.n_heads)
    for w_qkv, qkv in zip([model.W_Q, model.W_K, model.W_V], "qkv")
]

wi_weights = [
    (
        model.W_in[l].transpose(-2, -1),
        f"wi_L{l}",
        [
            f"wi_L{l}N{n}"
            for n in range(d_mlp)
        ],
    )
    for l in range(model.cfg.n_layers)
]

unembed_weights = [
    (model.W_U.T, "u", [f"u_{move_id_to_text(i, size)}" for i in all_squares]),
]

o_weights = [
    (
        model.W_O[l, h],
        f"o_L{l}H{h}",
        [
            f"o_L{l}H{h}D{d}"
            for d in range(d_head)
        ],
    )
    for l in range(model.cfg.n_layers)
    for h in range(model.cfg.n_heads)
]

wo_weights = [
    (
        model.W_out[l],
        f"wo_L{l}",
        [
            f"wo_L{l}N{n}"
            for n in range(d_mlp)
        ],
    )
    for l in range(model.cfg.n_layers)
]

in_weights = qkv_weights + wi_weights
out_weights = o_weights + wo_weights
all_weights = in_weights + out_weights
all_weights_WD = t.cat([w.flatten(0, -2) for w, _, _ in all_weights], dim=0)
all_weights_labels = sum((labels for _, _, labels in all_weights), [])
all_weights_groups = [group for _, group, _ in all_weights]
all_weights_indices = np.cumsum([0] + [len(labels) for _, _, labels in all_weights])
all_weights_WD.shape, len(all_weights_labels), len(all_weights_groups), len(all_weights_indices), all_weights_indices

# %%
z = (all_weights_WD / all_weights_WD.norm(dim=1, keepdim=True)) @ dict_DN
compressed_z = t.zeros(
    len(all_weights_groups),
    len(dict_probes),
    device=device,
)

for i in range(len(all_weights_groups)):
    for j in range(len(dict_probes)):
        group_z = z[
            all_weights_indices[i]:all_weights_indices[i+1],
            dict_indices[j]:dict_indices[j+1],
        ]
        x = group_z.abs().max().item()
        if x > 0.5:
            compressed_z[i, j] = x

fig = go.Figure(
    data=go.Heatmap(
        z=compressed_z.T.detach().cpu(),
        y=[k for k, _, _ in dict_probes],
        x=all_weights_groups,
        colorscale="gray",
        colorbar=dict(title="Value"),
    )
)
fig.update_layout(
    title="Heatmap of weights_WD @ dict_DN",
    yaxis_title="dict_DN Labels",
    xaxis_title="weights_WD Labels",
    xaxis=dict(tickangle=45),
)
fig.show()

# %%
for _weights in [
    # embed_weights + unembed_weights,
    qkv_weights[::3],
    qkv_weights[1::3],
    qkv_weights[2::3],
    o_weights,
    wi_weights,
    wo_weights,
    # [qkv_weights[60 * 3]],
    # [qkv_weights[60 * 3 + 1]],
    # [qkv_weights[60 * 3 + 2]],
    # [o_weights[60]],
    # wi_weights,
    # wo_weights,
]:
    weights_WD = t.cat([w.flatten(0, -2) for w, _, _ in _weights], dim=0)
    labels = sum((labels for _, _, labels in _weights), [])
    print(weights_WD.shape, len(labels))

    heatmap_data = weights_WD @ dict_DN

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.T.detach().cpu(),
            y=dict_labels,
            x=labels,
            colorscale="gray",
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        title="Heatmap of weights_WD @ dict_DN",
        yaxis_title="dict_DN Labels",
        xaxis_title="weights_WD Labels",
        xaxis=dict(tickangle=45),
    )
    fig.show()

# %%
def calculate_pairwise_colinearity(bases: Float[t.Tensor, "d_model basis"]):
    normalized_bases = bases / bases.norm(dim=0, keepdim=True)
    colinearity_matrix = t.matmul(normalized_bases.T, normalized_bases)
    return colinearity_matrix

threshold = 0.1
colinearity_matrix = calculate_pairwise_colinearity(dict_DN)
colinearity_matrix = t.tril(colinearity_matrix)
colinearity_matrix = t.where(
    colinearity_matrix.abs() > threshold, colinearity_matrix, 0
)
fig = go.Figure(
    data=go.Heatmap(
        z=colinearity_matrix.cpu().numpy(),
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        y=dict_labels,
        x=dict_labels,
    )
)
fig.update_yaxes(
    autorange="reversed",
)
fig.update_layout(
    title=f"Dict Probe Colinearity Heatmap (threshold {threshold})",
    xaxis=dict(title="Dict Vectors"),
    yaxis=dict(title="Dict Vectors"),
    height=len(dict_labels) * 8,
    width=len(dict_labels) * 8,
    shapes=[
        dict(
            type="line",
            x0=i - 0.5,
            y0=-0.5,
            x1=i - 0.5,
            y1=len(dict_labels) - 0.5,
            line=dict(color="black", width=1),
        )
        for i in dict_indices
    ]
    + [
        dict(
            type="line",
            x0=-0.5,
            y0=i - 0.5,
            x1=len(dict_labels) - 0.5,
            y1=i - 0.5,
            line=dict(color="black", width=1),
        )
        for i in dict_indices
    ],
)

fig.show()

# %%
