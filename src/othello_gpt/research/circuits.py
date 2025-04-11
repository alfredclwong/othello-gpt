# %%
import bisect
from collections import defaultdict
from enum import Enum, auto
from itertools import product
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch as t
from datasets import load_dataset
from jaxtyping import Float
from plotly.subplots import make_subplots
from torch.types import Tensor
from tqdm import tqdm

from othello_gpt.data.vis import move_id_to_text, plot_game
from othello_gpt.model.sae import SAE, SAEType
from othello_gpt.research.circuits_util import (
    load_saes,
    plot_evals,
)
from othello_gpt.research.targets import (
    # c_if_ne_target,
    captures_target,
    empty_target,
    flip_parity_target,
    # l_if_e_target,
    legality_target,
    move_target,
    p_target,
    prev_tem_target,
    # tm_target,
    theirs_empty_mine_target,
)
from othello_gpt.util import get_all_squares, load_model, load_probes

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

model_version = "600k"
model_name = f"awonga/othello-gpt-{model_version}"
model = load_model(device, model_name)
model.requires_grad_(False)
saes = load_saes(model, model_name, device)

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

root_dir = Path().cwd()  # .parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"
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

{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
n_test = 1024
batched_test_dataset = (
    test_dataset.select_columns(["input_ids"]).take(n_test).batch(128)
)
with t.inference_mode():
    eval_dicts, test_forward_dicts = zip(
        *[
            sae.evaluate(batched_test_dataset)
            for sae in tqdm(saes, "Calculating SAE metrics")
        ]
    )
    eval_dict = {k: [d[k] for d in eval_dicts] for k in eval_dicts[0]}
    test_forward_dicts = [
        {k: d[k].reshape(n_test, model.cfg.n_ctx, -1) for k in d}
        for d in test_forward_dicts
    ]

# %%
metrics = [
    "x_norm",
    "n_alive",
    "l0",
    "frac_active",
    "loss_recovered_zero_abl",
    "kl_div",
]
y_ranges = [[0, None], [0, None], [0, None], [0, None], [0.9, 1], [0, None]]
plot_evals(eval_dict, metrics, y_ranges, saes)

# %%
sae_alive_idxs = [
    (d["acts_post"] >= sae.cfg.dead_threshold).flatten(0, 1).any(0).nonzero().squeeze()
    for sae, d in zip(saes, test_forward_dicts)
]

w_ep = model.W_E_pos / model.W_E_pos.norm(dim=-1, keepdim=True)
l_e = [f"b_{move_id_to_text(i, size)}" for i in all_squares]
l_p = [f"p_{i}" for i in range(model.cfg.n_ctx)]
l_ep = l_e + l_p

in_latents = [w_ep]
out_latents = [w_ep]
latent_labels: list[str] = l_ep

for sae, test_forward_dict, alive_idxs in zip(saes, test_forward_dicts, sae_alive_idxs):
    layer = sae.cfg.out_hook_layer

    if sae.sae_type is SAEType.ATTN_Z:
        n_head = sae.model.cfg.n_heads
        z_h = sae.W_dec_normalized.reshape(sae.cfg.d_sae, n_head, sae.model.cfg.d_head)
        w_o = sae.model.W_O[layer]
        head_latents = einops.einsum(
            z_h,
            w_o,
            "d_sae n_head d_head, n_head d_head d_model -> d_sae n_head d_model",
        )

        # Norm such that the original z vectors' out projections have norm 1
        # i.e. head_latents.sum(1).norm(dim=-1) should be equal to 1 everywhere
        head_latents /= head_latents.sum(1, keepdim=True).norm(dim=-1, keepdim=True)

        # Append d_sae * n_head latents
        in_latents.append(z_h[alive_idxs].flatten(0, 1))
        out_latents.append(head_latents[alive_idxs].flatten(0, 1))
        latent_labels += [
            f"a{layer}f{i}h{j}"
            for i, j in product(range(len(alive_idxs)), range(n_head))
        ]

    elif sae.sae_type is SAEType.TRANSCODER:
        w_in = sae.W_enc.T
        w_in = w_in / w_in.norm(dim=-1, keepdim=True)
        in_latents.append(w_in[alive_idxs])
        out_latents.append(sae.W_dec_normalized[alive_idxs])
        latent_labels += [f"m{layer}f{i}" for i in range(len(alive_idxs))]

    # elif sae.sae_type is SAEType.LN_EMBED:
    #     in_latents.append(sae.W_dec_normalized[alive_idxs])
    #     out_latents.append(sae.W_dec_normalized[alive_idxs])
    #     latent_labels += [f"ln{layer}f{i}" for i in range(len(alive_idxs))]

    elif sae.sae_type in [SAEType.LN1, SAEType.LN2, SAEType.LN_FINAL]:
        w_in = sae.W_enc.T
        w_in = w_in / w_in.norm(dim=-1, keepdim=True)
        in_latents.append(w_in[alive_idxs])
        out_latents.append(sae.W_dec_normalized[alive_idxs])
        latent_labels += [
            f"{sae.sae_type.name}l{layer}f{i}" for i in range(len(alive_idxs))
        ]

    else:
        raise ValueError("Unrecognised sae type")

w_u = model.W_U.T
w_u = w_u / w_u.norm(dim=-1, keepdim=True)
in_latents.append(w_u)
out_latents.append(w_u)
latent_labels += [f"u_{move_id_to_text(i, size)}" for i in all_squares]

latent_idxs = t.tensor(
    np.cumsum([0] + [l.shape[0] for l in in_latents]), device=device
).tolist()
(
    len(latent_labels),
    [(x.shape, y.shape) for x, y in zip(in_latents, out_latents)],
    ", ".join(map(str, latent_idxs)),
    [latent_labels[i] for i in latent_idxs[:-1]],
)

# %%
# TODO diff with the average board state!
# Construct a priori auto-interps
# Weighted avg by acts_pre
# Outputs:
#   - board_states tensor
#       shape: (latent, state, row, col)
#       desc: act-weighted average binary board state, e.g. l/tm/e/bw/c/mov/parity
#   - pos tensor
#       shape: [latent, pos]
n_test_large = 10240
batch_size = 128
cols = ["legalities", "coords", "boards", "flips", "input_ids"]
large_batched_test_dataset = (
    test_dataset.select_columns(cols).take(n_test_large).batch(batch_size)
)

board_state_targets = {
    # "l": l_if_e_target,
    "l": legality_target,
    "e": empty_target,
    # "tm": tm_target,
    "tm": lambda x, d: theirs_empty_mine_target(x, d) / 2,
    "ptm": lambda x, d: prev_tem_target(x, d) / 2,
    # "c": c_if_ne_target,
    "c": captures_target,
    "fp": flip_parity_target,
    "mov": move_target,
    "p": p_target,
}

board_state_data_keys = ["acts_count", "acts_sum", "states_sum", "states_wsum"]
n_acts = [latent_idxs[1]] + [len(idxs) for idxs in sae_alive_idxs] + [model.cfg.d_vocab]
board_state_data_values = [
    [t.zeros((len(board_state_targets), n, size, size), device=device) for n in n_acts]
    for _ in board_state_data_keys
]
board_state_data = dict(zip(board_state_data_keys, board_state_data_values))

for i, batch in enumerate(tqdm(large_batched_test_dataset)):
    for j in range(len(n_acts)):
        if 0 < j <= len(saes):
            sae_idx = j - 1
            sae = saes[sae_idx]
            acts_type = "acts_post"
            with t.inference_mode():
                acts = sae.forward_dataset(
                    large_batched_test_dataset.select([i]).select_columns("input_ids"),
                    keys=[acts_type],
                )[acts_type].float()
                acts = acts[:, sae_alive_idxs[sae_idx]]
        else:
            input_ids = t.tensor(batch["input_ids"], device=device)[:, :-1]
            acts = t.zeros((*input_ids.shape, n_acts[j]), device=device)
            if j == 0:  # embed acts
                acts.scatter_(2, input_ids.unsqueeze(-1), 1)
                pos_idxs = t.arange(model.cfg.n_ctx, device=device)
                acts[:, pos_idxs, pos_idxs + model.cfg.d_vocab] = 1
            else:  # unembed acts
                # TODO decide which value to use for unembed acts
                # can be logits, relu(logits), probs, or probs > max_probs/2
                with t.inference_mode():
                    logits = model(input_ids, return_type="logits")
                probs = logits.softmax(-1)
                # max_probs = probs.max(dim=-1, keepdim=True)[0]
                # acts = (probs > (max_probs / 2)).float()

                # acts = t.nn.functional.relu(logits)

                acts = probs
            acts = acts.flatten(0, 1).float()

        for k, target_fn in enumerate(board_state_targets.values()):
            board_state = target_fn(batch, device).float()
            board_state = board_state.reshape(-1, size, size)  # * 2 - 1
            # board_state is (batch size size)
            # acts is (batch, d_sae)
            board_state_data["acts_sum"][j][k] += einops.einsum(
                acts,
                (~board_state.isnan()).float(),
                "batch_pos d_sae, batch_pos row col -> d_sae row col",
            )
            board_state_data["acts_count"][j][k] += einops.einsum(
                (acts > 0).float(),
                (~board_state.isnan()).float(),
                "batch_pos d_sae, batch_pos row col -> d_sae row col",
            )
            board_state_data["states_wsum"][j][k] += einops.einsum(
                acts,
                board_state.nan_to_num(0),
                "batch_pos d_sae, batch_pos row col -> d_sae row col",
            )
            board_state_data["states_sum"][j][k] += einops.einsum(
                (acts > 0).float(),
                board_state.nan_to_num(0),
                "batch_pos d_sae, batch_pos row col -> d_sae row col",
            )

weighted_avg_board_states = [
    s / a for s, a in zip(board_state_data["states_wsum"], board_state_data["acts_sum"])
]
avg_board_states = [
    s / a
    for s, a in zip(board_state_data["states_sum"], board_state_data["acts_count"])
]

# %%
type Node = tuple[int, int]


class NodeType(Enum):
    EMBED = auto()
    # LN_EMBED = auto()
    LN1 = auto()
    ATTN_Z = auto()
    LN2 = auto()
    TRANSCODER = auto()
    LN_FINAL = auto()
    UNEMBED = auto()


def get_node_type(n: Node, saes: list[SAE]) -> NodeType:
    if n[0] == 0:
        return NodeType.EMBED
    if n[0] == len(saes) + 1:
        return NodeType.UNEMBED

    sae: SAE = saes[n[0] - 1]
    st = sae.sae_type
    # if st is SAEType.LN_EMBED:
    #     return NodeType.LN_EMBED
    if st is SAEType.LN1:
        return NodeType.LN1
    if st is SAEType.ATTN_Z:
        return NodeType.ATTN_Z
    if st is SAEType.LN2:
        return NodeType.LN2
    if st is SAEType.TRANSCODER:
        return NodeType.TRANSCODER
    if st is SAEType.LN_FINAL:
        return NodeType.LN_FINAL

    raise ValueError("Unrecognised node type", n)


def node_to_latent_idx(n: Node, latent_idxs: list[int]) -> int:
    return latent_idxs[n[0]] + n[1]


def node_to_label(n: Node, latent_idxs: list[int], latent_labels: list[str]) -> str:
    latent_idx = node_to_latent_idx(n, latent_idxs)
    return latent_labels[latent_idx]


def latent_idx_to_node(idx: int, latent_idxs: list[int]) -> Node:
    layer = bisect.bisect_right(latent_idxs, idx) - 1
    offset = idx - latent_idxs[layer]
    return (layer, offset)


def display_auto_interp_data(nodes: list[Node], G=None):
    fig = make_subplots(
        rows=len(nodes),
        cols=len(board_state_targets),
        subplot_titles=list(board_state_targets.keys()) * len(nodes),
    )

    x_labels = [chr(97 + i) for i in range(size)]
    y_labels = [str(i + 1) for i in range(size)]

    for row_idx, n in enumerate(nodes, start=1):
        sae_idx = n[0] - 1
        latent_idx = n[1]
        if get_node_type(n, saes) == NodeType.ATTN_Z:
            latent_idx //= saes[sae_idx].model.cfg.n_heads

        for target_idx, target_label in enumerate(board_state_targets.keys()):
            weighted_avg_data = (
                weighted_avg_board_states[n[0]][target_idx, latent_idx].cpu().numpy()
            )

            scale = 1 / model.cfg.n_ctx if target_label == "p" else 1

            fig.add_trace(
                go.Heatmap(
                    z=weighted_avg_data,
                    colorscale="RdBu",
                    showscale=False,
                    zmin=0 * scale,
                    zmax=1 * scale,
                ),
                row=row_idx,
                col=target_idx + 1,
            )

            fig.update_xaxes(
                tickvals=list(range(size)),
                ticktext=x_labels,
                tickfont=dict(size=8),
                row=row_idx,
                col=target_idx + 1,
                showline=True,
                linecolor="black",
                linewidth=1,
                mirror=True,
                scaleanchor=f"y{row_idx}",
                scaleratio=1,
                constrain="domain",
            )
            fig.update_yaxes(
                tickvals=list(range(size)),
                ticktext=y_labels,
                tickfont=dict(size=8),
                row=row_idx,
                col=target_idx + 1,
                showline=True,
                linecolor="black",
                linewidth=1,
                mirror=True,
                constrain="domain",
                autorange="reversed",
            )

        # Add row label using node_to_label
        row_label = node_to_label(n, latent_idxs, latent_labels)
        parents = [(p, G[p][n]["weight"]) for p in G.predecessors(n)] if G else []
        parents_str = ", ".join(
            f"{node_to_label(p, latent_idxs, latent_labels)} ({w:.2f})"
            for p, w in parents
        )
        if parents_str:
            row_label += f" [{parents_str}]"
        ref = (row_idx - 1) * len(board_state_targets) + 1
        ref = "" if ref == 1 else ref
        fig.add_annotation(
            x=0,
            y=1.3,
            text=row_label,
            showarrow=False,
            font=dict(size=10, color="black"),
            xref=f"x{ref} domain",
            yref=f"y{ref} domain",
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        height=max(240, 120 * len(nodes)),
        width=100 * len(board_state_targets),
        title="Board states weighted by feature post_acts",
        margin=dict(l=10, r=10, t=80, b=10),
        showlegend=False,
        xaxis=dict(tickangle=0),
    )

    fig.show()


def display_probe_alignments(
    nodes: list[Node],
    G=None,
    probe_keys: None | list[str] = None,
    last_is_root: bool = True,
):
    if probe_keys is None:
        probe_keys = ["u", "l", "ee", "+t-m", "c", "mov"]

    node_latents = [out_latents[n[0]][n[1]] for n in nodes]
    node_latents = [x / x.norm() for x in node_latents]
    nl = list(zip(nodes, node_latents))
    last_is_root = (
        last_is_root and get_node_type(nodes[-1], saes) is not NodeType.UNEMBED
    )
    if last_is_root:
        n = nodes[-1]
        nl.insert(-1, (n, in_latents[n[0]][n[1]]))

    fig = make_subplots(
        rows=len(nl),
        cols=len(probe_keys),
        subplot_titles=probe_keys * len(nodes),
    )

    x_labels = [chr(97 + i) for i in range(size)]
    y_labels = [str(i + 1) for i in range(size)]

    for row_idx, (n, latent) in enumerate(nl, start=1):
        node_type = get_node_type(n, saes)
        if node_type is NodeType.EMBED:
            probe_layer = 0
        elif node_type is NodeType.UNEMBED:
            probe_layer = -1
        else:
            probe_layer = saes[n[0] - 1].cfg.out_hook_layer

        for probe_idx, probe_key in enumerate(probe_keys):
            probe = probes[probe_key][..., probe_layer]
            probe = probe.reshape(model.cfg.d_model, size, size)
            alignments = einops.einsum(
                latent,
                probe,
                "d_model, d_model row col -> row col",
            )

            fig.add_trace(
                go.Heatmap(
                    z=alignments.cpu(),
                    colorscale="RdBu",
                    showscale=False,
                    zmin=-1,
                    zmax=1,
                ),
                row=row_idx,
                col=probe_idx + 1,
            )

            fig.update_xaxes(
                tickvals=list(range(size)),
                ticktext=x_labels,
                tickfont=dict(size=8),
                row=row_idx,
                col=probe_idx + 1,
                showline=True,
                linecolor="black",
                linewidth=1,
                mirror=True,
                scaleanchor=f"y{row_idx}",
                scaleratio=1,
                constrain="domain",
            )
            fig.update_yaxes(
                tickvals=list(range(size)),
                ticktext=y_labels,
                tickfont=dict(size=8),
                row=row_idx,
                col=probe_idx + 1,
                showline=True,
                linecolor="black",
                linewidth=1,
                mirror=True,
                constrain="domain",
                autorange="reversed",
            )

        row_label = node_to_label(n, latent_idxs, latent_labels)
        parents = [(p, G[p][n]["weight"]) for p in G.predecessors(n)] if G else []
        parents_str = ", ".join(
            f"{node_to_label(p, latent_idxs, latent_labels)} ({w:.2f})"
            for p, w in parents
        )
        if last_is_root and n == nodes[-1]:
            row_label += " (in)" if row_idx == len(nodes) else " (out)"
        if parents_str:
            row_label += f" [{parents_str}]"
        ref = (row_idx - 1) * len(probe_keys) + 1
        ref = "" if ref == 1 else ref
        fig.add_annotation(
            x=0,
            y=1.3,
            text=row_label,
            showarrow=False,
            font=dict(size=10, color="black"),
            xref=f"x{ref} domain",
            yref=f"y{ref} domain",
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        height=120 * len(nl),
        width=100 * len(probe_keys),
        title="Probe x feature colinearities",
        margin=dict(l=10, r=10, t=80, b=10),
        showlegend=False,
        xaxis=dict(tickangle=0),
    )

    fig.show()


def get_upstream_nodes_qk(in_latent, upstream_latents, sae: SAE, head_idx, threshold):
    n_upstream = upstream_latents.shape[0]

    w_q = sae.model.W_Q[sae.cfg.in_hook_layer, head_idx]
    w_k = sae.model.W_K[sae.cfg.in_hook_layer, head_idx]
    ws = [w_q, w_k]
    qk_latents = t.cat([upstream_latents @ w for w in ws], dim=0)
    qk_latents = qk_latents / qk_latents.norm(dim=-1, keepdim=True)

    qk_alignments = qk_latents @ in_latent
    _, topk_qk_idxs = t.topk(qk_alignments, k)  # only pos alignments for attn

    nodes = [[], []]
    alignments = [[], []]
    for qk_idx in topk_qk_idxs.tolist():
        v = qk_alignments[qk_idx]
        if v.abs() < threshold:
            break
        qk, latent_idx = divmod(qk_idx, n_upstream)
        c = latent_idx_to_node(latent_idx, latent_idxs)
        nodes[qk].append(c)
        alignments[qk].append(v)
    return nodes, alignments


def get_w_v(n, saes):
    sae = saes[n[0] - 1]
    n_head = sae.model.cfg.n_heads
    head_idx = n[1] % n_head
    w_v = sae.model.W_V[sae.cfg.in_hook_layer, head_idx]
    return w_v


def get_upstream_nodes(
    n: Node, saes: list[SAE], k: int = 2, threshold: float = 0.5
) -> tuple[
    list[Node],
    list[float],
    list[list[Node], list[Node]],
    list[list[float], list[float]],
]:
    node_type = get_node_type(n, saes)

    if node_type is NodeType.EMBED:
        return [], [], [], []

    in_latent = in_latents[n[0]][n[1]]

    # Collate upstream latents
    # Filter upstream layers:
    #   For target attn_z, only the previous ln1 upstream is valid
    #   For target transcoder, only the previous ln2 upstream is valid
    #   For target ln, any non-ln upstream is valid
    ln_types = [NodeType.LN1, NodeType.LN2, NodeType.LN_FINAL]
    if node_type in ln_types:
        n0s = [
            n0
            for n0 in range(n[0])
            if get_node_type((n0, 0), saes) not in ln_types
        ]
    else:
        n0s = [n[0] - 1]
    upstream_idxs = np.cumsum([0] + [latent_idxs[n0 + 1] - latent_idxs[n0] for n0 in n0s])
    upstream_latents = t.cat([out_latents[n0] for n0 in n0s], dim=0)
    # upstream_latents = t.cat(out_latents[: n[0]], dim=0)

    # Apply v transform if necessary
    if node_type is NodeType.ATTN_Z:
        # Project upstream latents into V space
        sae = saes[n[0] - 1]
        n_head = sae.model.cfg.n_heads
        head_idx = n[1] % n_head

        qk_nodes, qk_alignments = get_upstream_nodes_qk(
            in_latent, upstream_latents, sae, head_idx, threshold
        )

        upstream_latents = upstream_latents @ get_w_v(n, saes)

    upstream_latents = upstream_latents / upstream_latents.norm(
        dim=-1, keepdim=True
    )  # shape: n_upstream d_latent

    # Find topk abs alignments
    upstream_alignments = upstream_latents @ in_latent
    if node_type is NodeType.ATTN_Z:
        _, topk_upstream_idxs = t.topk(
            upstream_alignments, k
        )  # only pos alignments for attn
    else:
        # split topk between pos and neg alignments
        _, topk_upstream_idxs_pos = t.topk(upstream_alignments, (k + 1) // 2)
        _, topk_upstream_idxs_neg = t.topk(upstream_alignments, k // 2, largest=False)
        topk_upstream_idxs = t.cat([topk_upstream_idxs_pos, topk_upstream_idxs_neg])

    nodes = []
    alignments = []
    for upstream_idx in topk_upstream_idxs.tolist():
        v = upstream_alignments[upstream_idx]
        if v.abs() < threshold:
            break

        # upstream_idx indexes upstream_latents, which is split into parts of len(latext_idxs[n0])
        # find idx of the n0s, then use this to idx latent_idxs
        n0_idx = np.searchsorted(upstream_idxs, upstream_idx, side="right") - 1
        latent_idx = upstream_idx + latent_idxs[n0s[n0_idx]] - upstream_idxs[n0_idx]
        c = latent_idx_to_node(latent_idx, latent_idxs)
        nodes.append(c)
        alignments.append(v.item())

    if node_type is NodeType.ATTN_Z:
        return nodes, alignments, qk_nodes, qk_alignments
    return nodes, alignments, [], []


def get_coactivation_matrix(nodes: list[Node]):
    # coacts[i, j] = p(j|i) = p(i,j)/p(i)
    # acts_count[i] = #(i)
    # coacts_count[i, j] = #(i,j)
    acts_count = t.zeros((len(nodes)), device=device)
    coacts_count = t.zeros((len(nodes), len(nodes)), device=device)

    n_test_large = 10240
    batch_size = 128
    batched_dataset = (
        test_dataset.select_columns("input_ids").take(n_test_large).batch(batch_size)
    )

    nodes_by_layer = defaultdict(list)
    for n in sorted(nodes):
        nodes_by_layer[n[0]].append(n)

    for i, batch in enumerate(
        tqdm(
            batched_dataset,
            f"Calculating coactivation statistics over {n_test_large} games",
        )
    ):
        batch_acts = []

        input_ids = t.tensor(batch["input_ids"], device=device)[:, :-1]

        for _, latent_idx in nodes_by_layer[0]:  # embeds
            if latent_idx < model.cfg.d_vocab:  # token embed
                batch_acts.append(input_ids.flatten() == latent_idx)
            else:  # pos embed
                pos_idx = latent_idx - model.cfg.d_vocab
                pos_acts = t.zeros_like(input_ids)
                pos_acts[:, pos_idx] = 1
                batch_acts.append(pos_acts.flatten())

        for j, sae in enumerate(saes):
            if not nodes_by_layer[j + 1]:
                continue

            acts_type = "acts_post"
            with t.inference_mode():
                acts = sae.forward_dataset(
                    large_batched_test_dataset.select([i]).select_columns("input_ids"),
                    keys=[acts_type],
                )[acts_type].float()[:, sae_alive_idxs[j]]

            n_head = sae.model.cfg.n_heads
            for n in nodes_by_layer[j + 1]:
                act_idx = n[1]
                if get_node_type(n, saes) == NodeType.ATTN_Z:
                    act_idx //= n_head
                batch_acts.append(acts[:, act_idx] > 0)

        if unembed_nodes := nodes_by_layer[len(saes) + 1]:
            with t.inference_mode():
                logits = model(input_ids, return_type="logits")
            probs = logits.softmax(-1)
            for _, latent_idx in unembed_nodes:
                unembed_acts = probs[..., latent_idx] > (probs.max(dim=-1)[0] / 2)
                batch_acts.append(unembed_acts.flatten())

        batch_acts = t.stack(batch_acts, dim=-1).float()  # shape: batch_pos node
        batch_coacts = einops.einsum(
            batch_acts, batch_acts, "batch_pos m, batch_pos n -> m n"
        )

        acts_count += batch_acts.sum(0)
        coacts_count += batch_coacts

    coacts = coacts_count / acts_count.unsqueeze(-1)
    perm = [sorted(nodes).index(node) for node in nodes]
    coacts = coacts[perm][:, perm]
    return coacts


def trace_circuit(
    root,
    k: int,
    thresholds: dict[NodeType, float],
    saes: list[SAE],
    pos_only=False,
    dfs=True,
    node_limit=100,
    max_depth=None,
):
    if pos_only:
        k *= 2
    G = nx.DiGraph()
    q = [(root, None, 0, 0)]
    while q and (node_limit is None or G.number_of_nodes() < node_limit):
        n, p, v, d = q.pop() if dfs else q.pop(0)
        G.add_node(n)
        if p is not None:
            G.add_edge(p, n, weight=v, abs_weight=abs(v))
        if max_depth is not None and d == max_depth:
            continue
        threshold = thresholds.get(get_node_type(n, saes), 0.1)
        cs, vs, _, _ = get_upstream_nodes(n, saes, k, threshold)
        for c, v in zip(cs, vs):
            if pos_only and v < 0:
                continue
            q.append((c, n, v, d + 1))

    # Ensure all nodes have a subset_key attribute for multipartite layouts
    for node in G.nodes:
        G.nodes[node]["subset_key"] = node[0]

    return G


def draw_graph(G, latent_idxs, latent_labels, figsize=(16, 48), linear_spacing=True):
    # Generate the layout
    pos = {}

    nodes_per_layer = defaultdict(list)
    for n in G.nodes:
        nodes_per_layer[n[0]].append(n)

    layer_sizes = [
        latent_idxs[i + 1] - latent_idxs[i] for i in range(len(latent_idxs) - 1)
    ]

    for layer, ns in nodes_per_layer.items():
        layer_size = layer_sizes[layer]
        for n in ns:
            if linear_spacing:
                pos[n] = (layer, 1 - n[1] / layer_size)
            else:
                pos[n] = (layer, 1 - (ns.index(n) + 1) / (len(ns) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={
            node: node_to_label(node, latent_idxs, latent_labels) for node in G.nodes
        },
        node_size=800,
        font_size=8,
        edge_color=[G[u][v]["weight"] for u, v in G.edges],
        edge_cmap=plt.cm.RdBu,
        edge_vmin=-1,
        edge_vmax=1,
        ax=ax,
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        rotate=False,
        font_size=6,  # Make edge labels smaller
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="none", facecolor="none"
        ),  # Transparent background
    )
    plt.show()


def highlight_positive(val):
    return "font-weight: bold; color: red" if val > 1e-2 else ""


def get_act_df(nodes: list[Node], game_idx, filter_inactive: bool = True):
    acts = []
    for n in nodes:
        if get_node_type(n, saes) == NodeType.EMBED:
            input_ids = t.tensor(test_dataset[game_idx]["input_ids"], device=device)[
                :-1
            ]
            if n[1] < model.cfg.d_vocab:  # token embed
                acts.append(input_ids == n[1])
            else:
                pos_act = t.zeros(model.cfg.n_ctx, device=device)
                pos_act[n[1] - model.cfg.d_vocab] = 1
                acts.append(pos_act)
        elif get_node_type(n, saes) == NodeType.UNEMBED:
            input_ids = t.tensor(test_dataset[game_idx]["input_ids"], device=device)[
                :-1
            ]
            with t.inference_mode():
                logits = model(input_ids, return_type="logits")[0]
            prob = logits.softmax(-1)[:, n[1]]
            # max_prob = prob.max(dim=-1, keepdim=True)[0]
            # logit_act = prob > (max_prob / 2).float()
            # acts.append(logit_act)
            # acts.append(logits[:, n[1]])
            acts.append(prob)
        else:
            sae_idx = n[0] - 1
            latent_idx = n[1]
            if get_node_type(n, saes) == NodeType.ATTN_Z:
                sae = saes[sae_idx]
                n_head = sae.model.cfg.n_heads
                latent_idx //= n_head
            acts.append(
                test_forward_dicts[sae_idx]["acts_pre"][game_idx, :, sae_alive_idxs[sae_idx][latent_idx]]
            )
    acts = t.stack(acts, dim=-1)

    df_index = 1 + np.arange(model.cfg.n_ctx)
    df_columns = [node_to_label(n, latent_idxs, latent_labels) for n in nodes]
    df = pd.DataFrame(acts.cpu(), index=df_index, columns=df_columns)
    if filter_inactive:
        df = df.loc[:, (df > 0).any()]  # Remove columns with no positive values
    styled_df = df.style.format("{:.2f}").map(highlight_positive)
    return styled_df


def get_game_acts(
    nodes: list[Node], act_type: str = "acts_post"
) -> Float[Tensor, "batch node"]:
    latent_idxs_by_sae_idx = defaultdict(set)
    for n in nodes:
        if 0 < n[0] <= len(saes):
            latent_idxs_by_sae_idx[n[0] - 1].add(n[1])
    acts = t.cat(
        [
            test_forward_dicts[sae_idx][act_type].sum(1)[:, list(_latent_idxs)]
            for sae_idx, _latent_idxs in latent_idxs_by_sae_idx.items()
        ],
        dim=-1,
    )
    return acts


def display_coactivations(nodes: list[Node]):
    n_nodes = len(nodes)
    coacts = get_coactivation_matrix(nodes)
    labels = [node_to_label(n, latent_idxs, latent_labels) for n in nodes]
    fig = go.Figure(
        data=go.Heatmap(
            z=coacts.cpu(),
            colorscale="RdBu",
            showscale=False,
            zmin=0,
            zmax=1,
        )
    )
    fig.update_xaxes(
        tickvals=list(range(n_nodes)),
        ticktext=labels,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
    )
    fig.update_yaxes(
        tickvals=list(range(n_nodes)),
        ticktext=labels,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        constrain="domain",
        autorange="reversed",
    )

    fig_size = max(400, n_nodes * 30)
    fig.update_layout(
        height=fig_size,
        width=fig_size,
        margin=dict(l=10, r=10, t=50, b=10),
        title="Co-activation Heatmap: C[i, j] = p(j|i)",
        xaxis=dict(title="Node j"),
        yaxis=dict(title="Node i"),
    )
    fig.show()


def display_colinearities(nodes: list[Node], last_is_root: bool = True):
    node_latents = [out_latents[n[0]][n[1]] for n in nodes]
    if last_is_root and get_node_type(nodes[-1], saes) is NodeType.ATTN_Z:
        w_v = get_w_v(nodes[-1])
        node_latents = [x @ w_v for x in node_latents]
    labels = [node_to_label(n, latent_idxs, latent_labels) for n in nodes]
    last_is_root = (
        last_is_root and get_node_type(nodes[-1], saes) is not NodeType.UNEMBED
    )
    if last_is_root:
        node_latents.insert(-1, in_latents[nodes[-1][0]][nodes[-1][1]])
        labels.insert(-1, labels[-1] + " (in)")
        labels[-1] += " (out)"
    node_latents = t.stack(node_latents, dim=0)
    colins = t.tril(node_latents @ node_latents.T).cpu()
    text = [
        [
            ""
            if x == 0 or i < len(labels) - 1 - int(last_is_root)
            else f"{int(x.round(decimals=2) * 100)}"
            for x in row
        ]
        for i, row in enumerate(colins)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=colins,
            colorscale="RdBu",
            showscale=False,
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 8, "color": "white"},
        )
    )
    fig.update_xaxes(
        tickvals=list(range(len(labels))),
        ticktext=labels,
        # tickfont=dict(size=4),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
    )
    fig.update_yaxes(
        tickvals=list(range(len(labels))),
        ticktext=labels,
        # tickfont=dict(size=4),
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        constrain="domain",
        autorange="reversed",
    )
    fig_size = max(400, len(labels) * 30)
    fig.update_layout(
        height=fig_size,
        width=fig_size,
        margin=dict(l=10, r=10, t=50, b=10),
        title="Colinearity Heatmap",
        xaxis=dict(title="Node j"),
        yaxis=dict(title="Node i"),
    )
    fig.show()


def display_node_interp(
    n: Node, saes, latent_idxs, latent_labels, k=8, threshold=0.5, levels=2
):
    # 1a. Show df of aligned upstream latents: alignment, necessary/sufficient co-activating stats, frac active
    # 1b. Same for downstream latents
    # 2. Show auto interp data for all relevant latents
    # 3. Show a max activating/unactivating game
    # TODO frac active

    print(f"Node {node_to_label(n, latent_idxs, latent_labels)} {n} interp dash")

    # 1a.
    upstream_nodes, upstream_alignments, qk_nodes, qk_alignments = get_upstream_nodes(
        n, saes, k=k, threshold=threshold
    )
    all_nodes = upstream_nodes + [n]

    # Create a directed graph with n as the root and upstream_nodes as the children
    H = nx.DiGraph()
    H.add_node(n)
    for child, alignment in zip(upstream_nodes, upstream_alignments):
        H.add_node(child)
        w = round(alignment, 2)
        H.add_edge(n, child, weight=w, abs_weight=abs(w))
    for node in H.nodes:
        H.nodes[node]["subset_key"] = node[0]
    draw_graph(H, latent_idxs, latent_labels, figsize=(6, 8), linear_spacing=False)

    display_coactivations(all_nodes)
    display_colinearities(all_nodes)

    if qk_nodes:
        print("QK cosine similarities")
        qk_strs = [
            f"{k} {node_to_label(n, latent_idxs, latent_labels)} {n} {v:.2f}"
            for k, ns, vs in zip("QK", qk_nodes, qk_alignments)
            for n, v in zip(ns, vs)
        ]
        print("\n".join(qk_strs))
        # display_colinearities(qk_nodes + [n])
        # TODO support colinearities of qk vectors and n_in
        # Coactivations would be intractable as they are across positions...?

    # 2.
    display_auto_interp_data(all_nodes, H)
    display_probe_alignments(all_nodes, H)

    # 3.
    game_acts = get_game_acts(H.nodes, act_type="acts_pre")
    root_idx = list(H.nodes).index(n)
    max_game_idx = game_acts[..., root_idx].argmax().item()
    min_game_idx = game_acts[..., root_idx].argmin().item()
    # max_game_idx = game_acts.flatten(1).sum(1).argmax().item()
    # min_game_idx = game_acts.flatten(1).sum(1).argmin().item()
    max_df = get_act_df(all_nodes, max_game_idx, filter_inactive=False)
    min_df = get_act_df(all_nodes, min_game_idx, filter_inactive=False)

    display(max_df)
    plot_game(test_dataset[max_game_idx], subplot_size=100, title="Max activating game")
    display(min_df)
    plot_game(test_dataset[min_game_idx], subplot_size=100, title="Min activating game")

# %%
k = 8  # expand by k times at each node
root = (len(latent_idxs) - 2, 0)  # A1
# root = (len(latent_idxs) - 3, 20)
# root = (len(latent_idxs) - 2, 19)  # F4
# root = (len(latent_idxs) - 2, 7)  # B2
thresholds = {
    NodeType.UNEMBED: 0.3,
    NodeType.LN_FINAL: 0.2,
    NodeType.LN2: 0.5,
    NodeType.LN1: 0.5,
    NodeType.TRANSCODER: 0.5,
    NodeType.ATTN_Z: 0.2,
}
G = trace_circuit(
    root, k, thresholds, saes, pos_only=True, dfs=False, node_limit=500, max_depth=None
)
draw_graph(G, latent_idxs, latent_labels)
# display_auto_interp_data(G.nodes, G)


# %%
root = (len(saes) + 1, 0)  # A1
# root = (len(saes) + 1, 7)  # B2
# root = (len(saes) + 1, 19)  # F4
# root = (len(saes), 225)
# root = (1, 443 * 8 + 6)
# root = latent_idx_to_node(latent_labels.index("m2f830"), latent_idxs)
display_node_interp(root, saes, latent_idxs, latent_labels, k=8, threshold=0.1)

# %%
in_latents[root[0]][root[1]]

# %%
# n0s = np.random.randint(len(latent_idxs) - 1, size=10)
# nodes = [(n0, np.random.randint(latent_idxs[n0 + 1] - latent_idxs[n0])) for n0 in n0s]
nodes = [(1, i) for i in range(latent_idxs[2] - latent_idxs[1])]
display_auto_interp_data(nodes[:10])
display_probe_alignments(nodes[:10], last_is_root=False)

# %%
# Record residual stream variance and mean before and after each layernorm using TransformerLens' cache
n_test = 1024
input_ids = t.tensor(test_dataset.take(128)["input_ids"], device=device)[:, :-1]
_, cache = model.run_with_cache(input_ids)
# {k: (cache[k].mean(-1).abs().max().item(), cache[k].std(-1).max().item()) for k in cache}
{k: cache[k].std(-1).std() for k in cache}

# %%
[sae.b_dec.abs().mean() for sae in saes]

# %%
