# %%
import itertools
from itertools import product
from pathlib import Path
from typing import List

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

from othello_gpt.data.vis import (
    plot_game,
    plot_probe_preds,
    plot_in_basis,
    move_id_to_text,
)
from othello_gpt.research.targets import (
    captures_target,
    legality_target,
    prev_tem_target,
    tm_target,
    l_if_e_target,
    c_if_ne_target,
    c_if_pm_target,
    c_if_t_target,
    ptm_target,
    empty_target,
    prev_empty_target,
)
from othello_gpt.util import (
    get_all_squares,
    load_model,
    load_probes,
    test_linear_probe,
)

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"
probe_dir.mkdir(parents=True, exist_ok=True)

# hf.login((root_dir / "secret.txt").read_text())
wandb.login()

size = 6
all_squares = get_all_squares(size)
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
model = load_model(device, "awonga/othello-gpt-6M")
model

# %%
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    w_p=model.W_pos.T.detach(),
    combos=[
        "+pee-ee",
        # "+b+ee",
        "+le+ee",
        "+cpm-ptm-pee+ee",  # TODO properly realign cpm to pm
        "+cpm-ptm",
    ],
)
{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
def project(
        xs: Float[t.Tensor, "d n"], bs: Float[t.Tensor, "d b"],
        norm=True, tol=1e-6, var_only=False,
):
    nbs, _, _ = t.svd(bs)
    nbs /= nbs.norm(dim=0, keepdim=True)
    dots = einops.einsum(
        xs, nbs,
        "d n, d b -> n b"
    )
    vars = (dots.norm(dim=1) / xs.norm(dim=0)) ** 2
    if var_only:
        return vars

    projected_xs = einops.einsum(
        dots, nbs,
        "n b, d b -> d n"
    )
    residual_xs = xs - projected_xs
    residual_xs = t.where(projected_xs.abs() < tol, 0, projected_xs)
    if norm:
        residual_xs /= residual_xs.norm(dim=0, keepdim=True).nan_to_num(1)
    return residual_xs, vars

# See how much of embed[y, x] is explained by +pe-ee[y, x] and tm[y, x]
for y in range(size):
    for x in range(size):
        i = y * size + x
        b = probes["b"][:, i, 0].unsqueeze(-1).clone()
        vs = []
        for k in ["+pee-ee", "tm", "ee"]:
            p = probes[k][:, i, 4].unsqueeze(-1)
            _, v = project(b, p)
            vs.append(v.item())
        print(y, x, [f"{v:.2f}" for v in vs])

# %%
# Probe W_E (I expect to see captures!)
labels = [move_id_to_text(i, size) for i in range(size * size)]
for k in ["+pee-ee", "cpm", "tm", "ee", "+cpm-ptm"]:
    if k in "dp":
        continue
    plot_in_basis(
        # project(
            # project(
        probes["b"][..., 0].T,
                # probes["+pe-ee"][..., 4],
                # probes["cne"][..., 4]
            # )[0],
            # probes["tm"][..., 4],
        # )[0].T,
        probes[k][..., 8],
        labels=labels,
        title=f"W_E in {k} probe basis",
        sort_by="",
        n_cols=size,
    )

# %%
# Probe W_P (I expect to see tm? e?)
for k in ["+pee-ee"]:#["tm", "ptm", "ee", "pee", "c", "le"]:
    if k in "dp":
        continue
    plot_in_basis(
        probes["p"][..., 0].T,
        probes[k][..., 0],
        labels=all_squares,
        title=f"W_P in {k} probe basis",
        sort_by="",
        n_cols=size,
    )

# %%
# Visualise cross-orthogonality between linear probes, across layers and features
# probe_keys = ["b", "+pe-ee", "ee", "tm", "le", "c", "cpm", "u", "+pm-pt", "pe"]
probe_keys = ["ee", "tm", "cpm", "ptm", "le", "+pee-ee", "+cpm-ptm", "c"]
dot_names, dot_probes = zip(*{k: probes[k] for k in probe_keys if k not in "dp"}.items())
dot_probes = t.stack(tuple(dot_probes), dim=-1)
probe_layers = [0, 1, -4, -1]
# probe_layers = list(range(dot_probes.shape[-2]))
dot_probes = dot_probes[:, all_squares][..., probe_layers, :]
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
positional_keys = ["ee", "tm", "cpm", "ptm", "le", "+pee-ee"]
# positional_keys = [k for k in probes.keys() if k not in "dp"]
board_probes = t.stack([probes[k] for k in positional_keys], dim=-2)
positional_dots = einops.einsum(
    board_probes[:, : board_probes.shape[1] // 2],
    board_probes,
    "d_model rc0 n_probe n_layer, d_model rc1 n_probe n_layer -> rc0 rc1 n_probe n_layer",
)
positional_dots = einops.reduce(
    positional_dots, "rc0 rc1 n_probe n_layer -> (rc0 n_probe) rc1", "mean"
).cpu()
positional_dots = einops.rearrange(positional_dots, "n (r c) -> n r c", r=size)
positional_dots = t.where(positional_dots >= 0.99, t.nan, positional_dots)
avg_pos_dots = (
    positional_dots.abs()
    .flatten(1)
    .nanmean(1)
    .reshape(-1, len(positional_keys))
    .nanmean(0)
    .tolist()
)
sorted_keys_and_dots = sorted(zip(positional_keys, avg_pos_dots), key=lambda x: x[1])
for key, dot in sorted_keys_and_dots:
    print(f"{key}: {dot}")
plot_game(
    {"boards": positional_dots},
    hovertext=positional_dots,
    reversed=False,
    subplot_titles=[
        f"{chr(ord('A') + x)}{y + 1} {p}"
        for y in range(size // 2)
        for x in range(size)
        for p in positional_keys
    ],
    shift_legalities=False,
    n_cols=len(positional_keys),
)

# %%
# Find % of residual stream variance explained by each probe direction
input_ids = t.tensor(test_dataset["input_ids"][:200], device=device)
_, cache = model.run_with_cache(input_ids[:, :-1])
X, y_labels = cache.get_full_resid_decomposition(
    apply_ln=True, return_labels=True, expand_neurons=False
)
X /= X.norm(dim=-1, keepdim=True)
X_cum, y_cum_labels = cache.accumulated_resid(
    apply_ln=True,
    return_labels=True,
    incl_mid=True,
)
X_cum /= X_cum.norm(dim=-1, keepdim=True)

corners = [0, size - 1, size * (size - 1), size * size - 1]
non_corners = [i for i in range(size * size) if i not in corners]
edges = [
    y * size + x
    for y in range(size)
    for x in range(size)
    if x * y == 0 or x == size - 1 or y == size - 1
]
non_edges = [i for i in range(size * size) if i not in edges]

# basis_keys = [
#     *[
#         (k, i, l)
#         for k, l, s in [
#             ("u", 0, all_squares),
#             # ("tm", 7, range(size * size)),
#         ] for i in s
#     ],
# ]
# basis_probes = {f"{k}_{move_id_to_text(i, size)}": probes[k][:, [i], l] for k, i, l in basis_keys}

# TODO why does removing pee bases increase cum %var? must be bug
basis_probes = {}
basis_probes["+cpm-ptm"] = probes["+cpm-ptm"][:, non_corners, 4]
# basis_probes["pee"] = probes["pee"][:, all_squares, 5]
basis_probes["ee"] = probes["ee"][:, all_squares, 5]
basis_probes["tm"] = probes["tm"][:, non_edges, 6]
basis_probes["ptm"] = probes["tm"][:, non_edges, 6]
basis_probes["+le+ee"] = probes["+le+ee"][:, all_squares, -2]
# basis_probes["u"] = probes["u"][:, all_squares, 0]
print({k: p.shape for k, p in basis_probes.items()})

probe_bases = t.cat(list(basis_probes.values()), dim=1)
probe_dims = {k: p.shape[1] for k, p in basis_probes.items()}
print(probe_dims)
print(X.shape, probe_bases.shape)

fig = make_subplots(
    rows=len(basis_probes),
    cols=2,
    subplot_titles=[f"{k} {x}" for k in basis_probes for x in ["decomp", "cum"]],
    vertical_spacing=0.5 / len(basis_probes),
)
probe_l = 0
for i, k in enumerate(basis_probes):
    probe_r = probe_l + probe_dims[k]
    U, _, _ = t.svd(probe_bases[:, probe_l:probe_r])
    d = einops.einsum(
        X,
        U,
        "layer batch pos d_model, d_model probe -> layer batch pos probe",
    )
    v = d.square().mean(1).sum(-1).detach().cpu()
    # print(k, v.nanmean().item(), (v.mean(1) * 100).int().tolist())
    # v = d.square().mean(1).sum(1).detach().cpu()
    fig.add_trace(
        go.Heatmap(
            z=v,
            y=y_labels,
            colorscale="gray",
            zmin=0,
            zmax=1.0,
            showscale=False,
        ),
        row=i + 1,
        col=1,
    )
    d_cum = einops.einsum(
        X_cum,
        U,
        "layer batch pos d_model, d_model probe -> layer batch pos probe",
    )
    v_cum = d_cum.square().mean(1).sum(-1).detach().cpu()
    fig.add_trace(
        go.Heatmap(
            z=v_cum,
            y=y_cum_labels,
            colorscale="gray",
            zmin=0,
            zmax=1.0,
            showscale=False,
        ),
        row=i + 1,
        col=2,
    )
    probe_l = probe_r
fig.update_layout(
    title_text="Variance Explained by Each Probe Direction",
    margin=dict(l=10, r=10, t=30, b=10),
    height=len(basis_probes) * 500,
)
fig.show()

probe_bases_orthogonal, _, _ = t.svd(probe_bases)
orthogonal_dots = einops.einsum(
    X,
    probe_bases_orthogonal,
    "layer batch pos d_model, d_model probe -> layer batch pos probe",
)
orthogonal_vars = orthogonal_dots.square().mean(1).sum(-1).detach().cpu()
orthogonal_cum_dots = einops.einsum(
    X_cum,
    probe_bases_orthogonal,
    "layer batch pos d_model, d_model probe -> layer batch pos probe",
)
orthogonal_cum_vars = orthogonal_cum_dots.square().mean(1).sum(-1).detach().cpu()
fig = make_subplots(rows=1, cols=2, subplot_titles=["decomp", "cum"])
fig.add_trace(
    go.Heatmap(
        z=orthogonal_vars,
        y=y_labels,
        colorscale="gray",
        zmin=0,
        zmax=1,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Heatmap(
        z=orthogonal_cum_vars,
        y=y_cum_labels,
        colorscale="gray",
        zmin=0,
        zmax=1,
    ),
    row=1,
    col=2,
)
fig.show()

# %%
plot_game(test_dataset[0])
for k in basis_probes:
    plot_in_basis(
        X_cum[-1, 0],
        probes[k][..., 6],
        list(range(model.cfg.n_ctx)),
        sort_by="",
        title=k,
        n_cols=8,
    )

# %%
linear_probe_ee = t.stack([-probes["ee"], probes["ee"]], dim=-2)
linear_probe_cap = t.stack([-probes["c"], probes["c"]], dim=-2)
# linear_probe_unembed = t.stack([-probes["u"], probes["u"]], dim=-2)  # this doesn't make any sense to do
linear_probe_tm = t.stack([-probes["tm"], probes["tm"]], dim=-2)
linear_probe_le = t.stack([-probes["le"], probes["le"]], dim=-2)
linear_probe_lee = t.stack([-probes["+le+ee"], probes["+le+ee"]], dim=-2)
linear_probe_cne = t.stack([-probes["cne"], probes["cne"]], dim=-2)
linear_probe_ct = t.stack([-probes["ct"], probes["ct"]], dim=-2)
linear_probe_cpm = t.stack([-probes["cpm"], probes["cpm"]], dim=-2)
linear_probe_ptm = t.stack([-probes["ptm"], probes["ptm"]], dim=-2)
linear_probe_pee = t.stack([-probes["pee"], probes["pee"]], dim=-2)
linear_probe_peeee = t.stack([-probes["+pee-ee"], probes["+pee-ee"]], dim=-2)

linear_probe_combo = t.stack(
    [
        -probes["+cpm-ptm"],
        probes["+cpm-ptm"],
    ],
    dim=-2,
)
# linear_probe_combo2 = t.stack(
#     [
#         -probes["+tm-cpm-cpm"],
#         probes["+tm-cpm-cpm"],
#     ],
#     dim=-2,
# )

pptem_target = lambda x, device: prev_tem_target(x, device, n_shift=2)

# %%
probe_targets = [
    # ("unembed (legal)", linear_probe_unembed, legality_target),
    # ("unembed (next move)", linear_probe_unembed, next_move_target),
    # ("tem", linear_probe_tem, theirs_empty_mine_target),
    # (
    #     "m",
    #     linear_probe_m,
    #     lambda x, device: (theirs_empty_mine_target(x, device) == 0).int(),
    # ),
    # (
    #     "e",
    #     linear_probe_e,
    #     lambda x, device: (theirs_empty_mine_target(x, device) == 1).int(),
    # ),
    # (
    #     "t",
    #     linear_probe_t,
    #     lambda x, device: (theirs_empty_mine_target(x, device) == 2).int(),
    # ),
    # ("pm", linear_probe_pm, lambda x, device: (prev_tem_target(x, device) == 2).int()),
    # ("pe", linear_probe_pe, lambda x, device: (prev_tem_target(x, device) == 1).int()),
    # ("pt", linear_probe_pt, lambda x, device: (prev_tem_target(x, device) == 0).int()),
    # ("cap", linear_probe_cap, captures_target),
    # ("legal (legal)", linear_probe_legal, legality_target),
    # ("legal (next move)", linear_probe_legal, next_move_target),
    # ("ptem", linear_probe_ptem, prev_tem_target),
    # ("pptem", linear_probe_pptem, pptem_target),
    # ("dir", linear_probe_dir, flip_dir_target),
    ("ee", linear_probe_ee, empty_target),
    ("tm|!e", linear_probe_tm, tm_target),
    ("le|e", linear_probe_le, l_if_e_target),
    ("le+ee", linear_probe_lee, legality_target),
    # ("cne|!e", linear_probe_cne, c_if_ne_target),
    # ("ct|t", linear_probe_ct, c_if_t_target),
    ("cpm|pm", linear_probe_cpm, c_if_pm_target),
    # ("bw", linear_probe_bw, bw_target),
    ("ptm|!pe", linear_probe_ptm, ptm_target),
    ("pee", linear_probe_pee, prev_empty_target),
    ("+cpm-ptm", linear_probe_combo, captures_target),
    # ("combo2", linear_probe_combo2, ptm_target),
]

probe_accuracies = {}
probe_losses = {}
labels = []
for name, probe, target_fn in tqdm(probe_targets):
    test_y: Float[t.Tensor, "n_test pos n_out"] = target_fn(test_dataset, device)
    test_loss, test_accs, labels = test_linear_probe(
        model,
        device,
        test_dataset,
        test_y,
        probe,
        target_fn=lambda x: target_fn(x, device),
        scalar_loss=False,
        return_labels=True,
    )

    test_loss = einops.reduce(
        test_loss, "layer batch pos n_out -> layer pos n_out", t.nanmean
    )
    test_accs = einops.reduce(
        test_accs.float(), "layer batch pos n_out -> layer pos n_out", t.nanmean
    )

    probe_accuracies[name] = test_accs
    probe_losses[name] = test_loss.detach().cpu()

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

# %%
fig_loss_layer = make_subplots(
    rows=1, cols=2, subplot_titles=("Test Loss", "Test Accuracy")
)

for name in probe_losses:
    fig_loss_layer.add_trace(
        go.Scatter(
            x=labels,
            y=probe_losses[name].flatten(1).nanmean(dim=1),
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
            x=labels,
            y=probe_accuracies[name].flatten(1).nanmean(dim=1),
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
batch = dataset_dict["test"].take(1)
plot_game(batch[0])
# TODO test_loss.argmax() for probe_layer
preds = [
    # (linear_probe_cap, captures_target, "Captures probe", 4),
    (linear_probe_ptm, ptm_target, "PT-PM probe", -6),
    (linear_probe_tm, tm_target, "T-M probe", -6),
    (linear_probe_ee, empty_target, "EE probe", 3),
    (linear_probe_le, l_if_e_target, "L if E probe", -2),
    (linear_probe_combo, captures_target, "C|PM - PTM - (PE - E) probe", 4),
    # (linear_probe_combo2, ptm_target, "Combo2 probe", -2),

    # (linear_probe_cne, c_if_ne_target, "C if not E probe", 4),
    # (linear_probe_cne, c_if_ne_target, "C if not E probe L0", 0),
    # (linear_probe_ct, c_if_t_target, "C if T probe", 4),
    # (linear_probe_ct, c_if_t_target, "C if T probe L0", 0),
    # (linear_probe_cpm, c_if_pm_target, "C if PM probe", 4),
    # (linear_probe_cpm, c_if_pm_target, "C if PM probe L0", 0),
    # (linear_probe_cpm, c_if_pm_target, "C if PM probe L1", 1),

    # *[
    #     (linear_probe_tm, tm_target, f"T-M probe L{i}", i)
    #     for i in range(linear_probe_tm.shape[-1])
    # ],
    # *[
    #     (linear_probe_peeee, empty_target, f"+PEE-EE probe L{i}", i)
    #     for i in range(linear_probe_tm.shape[-1])
    # ],
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
# Show the evolution of preds for a fixed probe across acc_resid
batch = dataset_dict["test"].take(1)
target = tm_target(batch, device).detach().cpu()
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
    linear_probe_tm,
    "l0 batch n_ctx d_model, d_model n_out d_probe l1 -> l0 l1 batch n_ctx n_out d_probe",
)
log_probs = preds.log_softmax(-1)
pred_prob = t.exp(log_probs).cpu()
pred_prob, pred_index = pred_prob.max(dim=-1)
pred_index = pred_index.float()
pred_index[..., t.isnan(target)] = t.nan

n_out = probe.shape[1]
if n_out < t.tensor(batch["moves"]).max():
    annotate_moves = False
else:
    annotate_moves = True

int_sqrt = int(n_out ** 0.5)
target = einops.rearrange(
    target, "batch pos (row col) -> batch pos row col", row=int_sqrt
)
pattern = "l0 l1 batch n_ctx (row col) -> l0 l1 batch n_ctx row col"
pred_prob = einops.rearrange(pred_prob, pattern, row=int_sqrt)
pred_index = einops.rearrange(pred_index, pattern, row=int_sqrt)

index = 0
for cache_layer in range(X.shape[0]):
    pred_dict = {
        "boards": pred_index[cache_layer, 5, index],
        "legalities": target[index] == 1,
        "moves": batch["moves"][index],
    }
    plot_game(
        pred_dict,
        reversed=False,
        textcolor="red",
        hovertext=pred_prob[cache_layer, 5, index],
        shift_legalities=False,
        title=f"TM5 pred on X{cache_layer}",
        annotate_moves=annotate_moves,
    )

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
                    y=probe_losses[name][..., y * size + x].nanmean(1),
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
                    y=probe_accuracies[name][..., y * size + x].nanmean(1),
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

# fig_loss.show()
fig_accs.show()

# Why is loss lower and acc higher for legal vs unembed on same legality target?
# Legal probe minimises cross-entropy for whether a specific square is legal, given a residual vector
# Unembed (transformer) mininmises cross-entropy over the next token

# %%
# # TODO subplot per probe, line per pos, x-axis layer, y-axis test/loss
# fig_probe_loss = make_subplots(
#     rows=len(probe_targets),
#     cols=1,
#     shared_xaxes=True,
#     subplot_titles=[name for name, _, _ in probe_targets],
# )
# fig_probe_acc = make_subplots(
#     rows=len(probe_targets),
#     cols=1,
#     shared_xaxes=True,
#     subplot_titles=[name for name, _, _ in probe_targets],
# )

# for i, (name, probe, target_fn) in enumerate(probe_targets):
#     for pos in range(model.cfg.n_ctx):
#         fig_probe_loss.add_trace(
#             go.Scatter(
#                 x=list(range(probe_losses[name].shape[0])),
#                 y=probe_losses[name][:, pos].nanmean(dim=1),
#                 mode="lines",
#                 name=f"{name} pos {pos}",
#                 line=dict(color=cmap[name]),
#                 showlegend=False,
#             ),
#             row=i + 1,
#             col=1,
#         )
#         fig_probe_acc.add_trace(
#             go.Scatter(
#                 x=list(range(probe_accuracies[name].shape[0])),
#                 y=probe_accuracies[name][:, pos].nanmean(dim=1),
#                 mode="lines",
#                 name=f"{name} pos {pos}",
#                 line=dict(color=f"rgba(0, 0, 255, {pos / model.cfg.n_ctx})"),
#                 showlegend=False,
#             ),
#             row=i + 1,
#             col=1,
#         )

# fig_probe_loss.update_layout(
#     height=300 * len(probe_targets),
#     width=1200,
#     title_text="Test Loss per Probe and Position",
# )
# fig_probe_acc.update_layout(
#     height=300 * len(probe_targets),
#     width=1200,
#     title_text="Test Accuracy per Probe and Position",
# )

# # fig_probe_loss.show()
# fig_probe_acc.show()

# %%
# Per probe loss/acc (z) across layers (x) and pos (y)
fig_probe_loss_acc = make_subplots(
    rows=len(probe_targets),
    cols=2,
    shared_xaxes=True,
    subplot_titles=[
        f"{title} {metric}"
        for title, _, _ in probe_targets
        for metric in ["loss", "acc"]
    ],
)

for i, (name, probe, target_fn) in enumerate(probe_targets):
    fig_probe_loss_acc.add_trace(
        go.Heatmap(
            z=probe_losses[name].nanmean(-1).T,
            y=list(range(model.cfg.n_ctx)),
            x=list(range(probe.shape[-1])),
            colorscale="Greys",
            showscale=False,
        ),
        row=i + 1,
        col=1,
    )
    fig_probe_loss_acc.add_trace(
        go.Heatmap(
            z=probe_accuracies[name].nanmean(-1).T,
            y=list(range(model.cfg.n_ctx)),
            x=list(range(probe.shape[-1])),
            colorscale="Greys_r",
            showscale=False,
        ),
        row=i + 1,
        col=2,
    )

fig_probe_loss_acc.update_layout(
    height=300 * len(probe_targets),
    width=1200,
    title_text="Test Loss and Accuracy per Probe and Position (Heatmap)",
)

fig_probe_loss_acc.show()

# %%
fig_loss = make_subplots(
    rows=size,
    cols=size,
    y_title="loss",
    x_title="layer",
    subplot_titles=range(model.cfg.n_ctx),
)
fig_accs = make_subplots(
    rows=size,
    cols=size,
    y_title="acc",
    x_title="layer",
    subplot_titles=range(model.cfg.n_ctx),
)

for i in range(model.cfg.n_ctx):
    y, x = divmod(i, size)
    showlegend = i == 0
    for name in probe_losses:
        fig_loss.add_trace(
            go.Scatter(
                y=probe_losses[name][:, i].nanmean(1),
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
                y=probe_accuracies[name][:, i].nanmean(1),
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
    height=fig_size, width=fig_size, title_text="Test Loss for Each Pos"
)
fig_accs.update_layout(
    height=fig_size, width=fig_size, title_text="Test Accuracy for Each Pos"
)
fig_accs.update_yaxes(range=[0.5, 1], row="all", col="all")

# fig_loss.show()
fig_accs.show()

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
