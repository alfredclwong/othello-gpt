# %%
import torch as t
from datasets import load_dataset
import huggingface_hub as hf
from pathlib import Path
import einops
import plotly.graph_objects as go
from typing import Union, List, Optional
from jaxtyping import Float
from transformer_lens import ActivationCache
import circuitsvis as cv
from IPython.display import HTML
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import numpy as np

from othello_gpt.data.vis import plot_in_basis, plot_game
from othello_gpt.util import (
    get_all_squares,
    load_model,
    load_probes,
    vocab_to_board,
)
from othello_gpt.data.vis import move_id_to_text

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"

# hf.login((root_dir / "secret.txt").read_text())
dataset_dict = load_dataset("awonga/othello-gpt")

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

size = 6
all_squares = get_all_squares(size)

# %%
version = "6M"
model = load_model(device, f"awonga/othello-gpt-{version}")
n_layer = model.cfg.n_layers
n_head = model.cfg.n_heads
d_head = model.cfg.d_head
d_model = model.cfg.d_model
n_neuron = model.cfg.d_model * 4

# %%
n_test = 200
test_dataset = dataset_dict["test"].take(n_test)

padded_W_pos = t.full((size * size, model.W_pos.shape[1]), t.nan, device=device)
padded_W_pos[: model.W_pos.shape[0], :] = model.W_pos
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    w_p=padded_W_pos.T.detach(),
    combos=["+t-m", "+pee-ee"],
    model_version=version,
)
probes["r"] = t.randn_like(probes["ee"])
probes["zs"] = t.stack([
    -probes["pos"][..., 0, :],
    -probes["pos"][..., 0, :],
    probes["pos"][..., 0, :],
    probes["pos"][..., 0, :],
], dim=-2)
probes["z"] = t.stack([
    -probes["pos"][..., 0, :],
    probes["pos"][..., 0, :],
], dim=-2)
{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
def visualize_attention_patterns(
    heads: Union[List[int], int, Float[t.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: t.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[t.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = [move_id_to_text(t, size) for t in local_tokens]

    # Combine the patterns into a single tensor
    patterns: Float[t.Tensor, "head_index dest_pos src_pos"] = t.stack(
        patterns, dim=0
    ).cpu()

    # Normalise relative to 1/pos such that later rows don't get diluted
    patterns *= (t.arange(patterns.shape[1]) + 1).unsqueeze(0).unsqueeze(-1)

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = cv.circuitsvis.attention.attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"

for i in range(3):
    test_game = test_dataset[i]
    test_input_ids = t.tensor(test_game["input_ids"], device=device)
    test_logits, test_cache = model.run_with_cache(test_input_ids[:-1])
    vis = visualize_attention_patterns(
        list(range(model.cfg.n_layers * model.cfg.n_heads)),
        test_cache,
        test_game["moves"],
    )
    display(HTML(vis))
    fig = plot_game(test_game, return_fig=True)
    fig.show()

# %%
input_ids = t.tensor(test_dataset[0]["input_ids"], device=device)
_, cache = model.run_with_cache(input_ids[:-1])
block = "blocks.0.ln2.hook_normalized"
x = cache[block][0]

labels = [f"{i+1}. {s}" for i, s in enumerate(test_dataset[0]["squares"])]
plot_game(test_dataset[0])
plot_in_basis(
    x,
    probes["mov"][..., 4],
    labels,
    n_cols=8,
    title=f"{block} in (MOV) basis",
)
plot_in_basis(
    x,
    padded_W_pos.T.detach(),
    labels,
    n_cols=8,
    title=f"{block} in (P) basis",
)
# plot_in_basis(
#     x,
#     probes["zs"][..., 4],
#     labels,
#     n_cols=8,
#     title="Z",
# )

# %%
plot_in_basis(
    probes["pos"][..., 0].T,
    probes["p"][..., 0],
    labels=[f"(pos)_{i}.(p)" for i in range(4)],
    title="(POS) in (P) basis",
    n_cols=4,
)

# %%
test_game = test_dataset[1]
input_ids = t.tensor(test_game["input_ids"], device=device)
_, cache = model.run_with_cache(input_ids[:-1])
x = cache["blocks.2.ln1.hook_normalized"][0]

all_square_labels = [move_id_to_text(i, size) for i in all_squares]
pos_labels = list(range(model.cfg.n_ctx))

circuits = {
    "O": (model.W_O.transpose(-2, -1), t.zeros_like(model.b_V)),
    "V": (model.W_V, model.b_V),
    "Q": (model.W_Q, model.b_Q),
    "K": (model.W_K, model.b_K),
}
layer = 2
head = 5
circuit = "QK"

bilinear_probes_desc = [
    ("z", slice(None)),
    ("z", slice(None)),
    # ("r", all_squares),
    # ("x", slice(None)),
    # ("ee", all_squares),
    # ("mov", all_squares),
]
moves = test_game["squares"][:-1]
labels = [
    ["~(Z)", "(Z)"],
    ["~(Z)", "(Z)"],
    # list(range(len(all_square_labels))),
    # moves,
    # pos_labels,
    # all_square_labels,
    # all_square_labels,
    # moves,
]
bilinear_probes = [
    x.T.clone() if k == "x" else
    probes[k][..., squares, 2 * layer].clone()
    for k, squares in bilinear_probes_desc
]
bilinear_probes[0] *= 10
bilinear_probes_desc[0] = ("10z", slice(None))
bilinear_probes[1] *= 10
bilinear_probes_desc[1] = ("10z", slice(None))
# bilinear_probes[0] = x.T
# bilinear_probes[1] = x.T

z = (
    (
        bilinear_probes[0].T @  # n_probe, d_model
        circuits[circuit[0]][0][layer, head]  # d_model, d_head
        + circuits[circuit[0]][1][layer, head]  # d_head
    ) @ (
        bilinear_probes[1].T @
        circuits[circuit[1]][0][layer, head]
        + circuits[circuit[1]][1][layer, head]
    ).T / np.sqrt(model.cfg.d_model)
).detach().cpu()
# z = t.masked_fill(z, t.triu(t.ones_like(z, dtype=bool), 1), -t.inf)
# z = z.softmax(1)
# z = (
#     bilinear_probes[0].T @ model.QK[layer, head] @ bilinear_probes[1]
# ).AB.detach().cpu()

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=z,
        y=labels[0],
        x=labels[1],
        colorscale="gray",
        showscale=False,
        xgap=0.2,
        ygap=0.2,
        texttemplate="%{text}",
    ),
)

fig.update_yaxes(
    showline=True,
    linecolor="black",
    linewidth=1,
    mirror=True,
    constrain="domain",
    autorange="reversed",
    title=f"{bilinear_probes_desc[0][0]} ({circuit[0]})",
    tickmode="array",
    tickvals=list(range(len(labels[0]))),
    ticktext=labels[0],
    tickfont=dict(size=8),
)

fig.update_xaxes(
    showline=True,
    linecolor="black",
    linewidth=1,
    mirror=True,
    scaleanchor="y",
    scaleratio=1,
    constrain="domain",
    title=f"{bilinear_probes_desc[1][0]} ({circuit[1]})",
    tickmode="array",
    tickvals=list(range(len(labels[1]))),
    ticktext=labels[1],
    tickfont=dict(size=8),
    tickangle=0,
)

fig.update_layout(
    title=f"L2H5 {bilinear_probes_desc[0][0]}.{circuit}.{bilinear_probes_desc[1][0]}",
    margin=dict(l=10, r=10, t=50, b=10),
    width=200,
    height=len(labels[0]) * 15 + 100,
)

fig.show()

# %%
d_head_labels = [f"d_head_{i}" for i in range(model.cfg.d_head)]
plot_in_basis(
    model.W_K[2, 5].T.detach(),
    probes["mov"][..., 4],
    labels=d_head_labels,
    title="L2H5 W_K.mov",
    n_cols=4,
    bias=model.b_K[2, 5].detach().cpu(),
)
plot_in_basis(
    model.W_K[2, 5].T.detach(),
    probes["zs"][..., 4],
    labels=d_head_labels,
    title="L2H5 W_K.z",
    n_cols=4,
    bias=model.b_K[2, 5].detach().cpu(),
)
plot_in_basis(
    model.W_Q[2, 5].T.detach(),
    probes["zs"][..., 0],
    labels=d_head_labels,
    title="L2H5 W_Q.z",
    n_cols=4,
    bias=model.b_Q[2, 5].detach().cpu(),
)

# %%
d_head_labels = [f"d_head_{i}" for i in range(model.cfg.d_head)]
plot_in_basis(
    model.W_O[2, 5].detach(),
    probes["ee"][..., 4],
    labels=d_head_labels,
    title="L2H5 W_O.ee",
    n_cols=4,
    bias=model.b_O[2, 5].detach().cpu(),
)
plot_in_basis(
    model.W_V[2, 5].T.detach(),
    probes["z"][..., 4],
    labels=d_head_labels,
    title="L2H5 W_V.(pee-ee)",
    n_cols=4,
    bias=model.b_V[2, 5].detach().cpu(),
)
plot_in_basis(
    model.W_V[2, 5].T.detach(),
    probes["mov"][..., 4],
    labels=d_head_labels,
    title="L2H5 W_V.mov",
    n_cols=4,
    bias=model.b_V[2, 5].detach().cpu(),
)

# %%
focus_batch = dataset_dict["test"].take(100)
focus_input_ids = t.tensor(focus_batch["input_ids"], device=device)[:, :-1]
_, focus_cache = model.run_with_cache(
    focus_input_ids,
    names_filter="blocks.2.attn.hook_pattern",
)
focus_cache




# %%
# Maybe there are some memory heads: if a move is played at a certain pos,
# this can imply that a certain opening was played! E.g. white plays E2 at
# move 2, this is only possible if black played D2 or E3 at move 1. If D2,
# then [F2, E3, B4, C5] are legal. If E3, then D2 instead of E3 is legal.
# So we query (E2, move 2), key (move 1), get value (PE-E for which move
# was played), and output (M-L).

# How do we find this circuit?
# 1. W_Q (E2, move 2)
# 2. W_K (move 1) or (D2+E3)
# 3. W_O (D2-E3)
# 4. W_V (M-L)

# %%
# Sort Q by E2, move 2 activation
e2_id = 10
e2_pos = 1
print(move_id_to_text(e2_id, size))
probe_layer = 10
probe = probes["+pee-ee"][:, e2_id, probe_layer]
probe += probes["p"][:, e2_pos, probe_layer]
probe = probe.unsqueeze(-1)

qs = model.W_Q.transpose(-1, -2).flatten(0, 2)
labels = [
    f"L{l}H{h}D{d}"
    for l in range(n_layer)
    for h in range(n_head)
    for d in range(d_head)
]
plot_in_basis(
    qs.detach().cpu(), probe.detach().cpu(), labels,
    sort_by="absmean", top_n=20,
)

# %%
plot_in_basis(
    model.W_Q[5, 1].T.detach().cpu(),
    probes["+pee-ee"][..., probe_layer].cpu(),
    labels=list(range(32)),
    title="L5H1 Q.(pe-e)",
)  # D[0,2,14,16,18]
plot_in_basis(
    model.W_Q[5, 1].T.detach().cpu(),
    p_padded,
    labels=list(range(32)),
    title="L5H1 Q.p",
)  # D[0,1,2!,9,18!]
plot_in_basis(
    model.W_K[5, 1].T.detach().cpu(),
    probes["+pee-ee"][..., probe_layer].cpu(),
    labels=list(range(32)),
    title="L5H1 K.(pe-e)",
)  # D[2!,8,9,16,18!20,21,26,27,29,30]
plot_in_basis(
    model.W_K[5, 1].T.detach().cpu(),
    p_padded,
    labels=list(range(32)),
    title="L5H1 K.p",
)  # :-(

# %%
# Sort V by +-(D2-E3), move 1 activation
d2_id = 9
e3_id = 14
d2_e3_pos = 0
print(move_id_to_text(d2_id, size), move_id_to_text(e3_id + 2, size))
probe = probes["+pee-ee"][:, d2_id, probe_layer]
probe -= probes["+pee-ee"][:, e3_id, probe_layer]
# probe += probes["p"][:, d2_e3_pos, probe_layer]
probe = probe.unsqueeze(-1)

os = model.W_O.flatten(0, 2)
vs = model.W_V.transpose(-1, -2).flatten(0, 2)
labels = [
    f"L{l}H{h}D{d}"
    for l in range(n_layer)
    for h in range(n_head)
    for d in range(d_head)
]
plot_in_basis(
    os.detach().cpu(), probe.detach().cpu(), labels,
    sort_by="absmean", top_n=20,
)
plot_in_basis(
    vs.detach().cpu(), probe.detach().cpu(), labels,
    sort_by="absmean", top_n=20,
)

# %%
plot_in_basis(
    model.W_V[5, 1].T.detach().cpu(),
    probes["+pee-ee"][..., probe_layer].cpu(),
    labels=list(range(32)),
)
plot_in_basis(
    model.W_V[5, 1].T.detach().cpu(),
    p_padded,
    labels=list(range(32)),
)

# %%
plot_in_basis(
    model.W_O[5, 1].detach().cpu(),
    probes["le"][..., probe_layer].cpu(),
    labels=list(range(32)),
)  # F2, B4, C5, D2/E3
plot_in_basis(
    model.W_O[5, 1].detach().cpu(),
    probes["tm"][..., probe_layer].cpu(),
    labels=list(range(32)),
)  # D2/E2
