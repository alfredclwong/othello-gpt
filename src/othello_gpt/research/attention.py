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
version = "4M"
model = load_model(device, f"awonga/othello-gpt-{version}")
n_layer = model.cfg.n_layers
n_head = model.cfg.n_heads
d_head = model.cfg.d_head
d_model = model.cfg.d_model
n_neuron = model.cfg.d_model * 4

# %%
n_test = 100
test_dataset = dataset_dict["test"].take(n_test)

probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    w_p=model.W_pos.T.detach(),
    # combos=["t+m", "t-m", "t-e", "t-pt", "m-pm"],
    combos=["+pee-ee"],
    model_version=version,
)
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

# %%
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
(
    model.W_Q.shape,
    model.W_K.shape,
    model.W_V.shape,
    model.W_O.shape,
)  # n_layer n_head d_model d_head

# %%
# L2H5 attends to D5 strongly and pos 0 weakly, dst invariant
# lh = (2, 5)
lh = (1, 0)
p = probes["tm"][:, all_squares, 4].T
# for i in [0, 1, 6, 7]:
for i in [2]:
    test_game = test_dataset[i]
    input_ids = t.tensor(test_game["input_ids"])
    moves = test_game["squares"][:-1]

    _, cache = model.run_with_cache(input_ids[:-1])
    x = cache[f"blocks.{lh[0]}.ln1.hook_normalized"][0]
    a = (x @ model.QK[*lh] @ x.transpose(-1, -2)).AB
    v = (p @ model.OV[*lh] @ x.transpose(-1, -2)).AB
    # Apply an attn mask to a
    attn_mask = t.tril(t.ones(a.shape[-2:], device=a.device, dtype=bool))
    a = t.where(attn_mask, a, -t.inf)

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Heatmap(
            z=a.softmax(1).detach().cpu(),
            # z=a.detach().cpu(),
            # z=cache[f"blocks.{lh[0]}.attn.hook_pattern"][0, lh[1]].detach().cpu(),
            # z=cache[f"blocks.{lh[0]}.attn.hook_attn_scores"][0, lh[1]].detach().cpu(),
            y=moves,
            x=moves,
            colorscale="gray",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=v.detach().cpu(),
            y=[move_id_to_text(i, size) for i in all_squares],
            x=moves,
            colorscale="gray",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        height=700,
        width=1600,
    )
    fig.update_yaxes(
        title_text="dst (Attention)",
        row=1, col=1,
    )
    fig.update_xaxes(
        title_text="src (Attention)",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="probe",
        row=1, col=2,
    )
    fig.update_xaxes(
        title_text="src",
        row=1, col=2,
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
    fig.show()

# %%
p_padded = probes["p"][..., 0]
p_padded = t.nn.functional.pad(
    p_padded, (0, size * size - p_padded.shape[1]), value=t.nan
)
p_padded = p_padded.detach()

b_padded = probes["b"][..., 0]
b_padded = t.nn.functional.pad(
    b_padded, (0, size * size - b_padded.shape[1]), value=t.nan
)
b_padded = b_padded.detach()

# %%
input_ids = t.tensor(test_dataset[0]["input_ids"], device=device)
_, cache = model.run_with_cache(input_ids[:-1])
x = cache["blocks.2.ln1.hook_normalized"][0]

labels = [f"{i+1}. {s}" for i, s in enumerate(test_dataset[0]["squares"])]
plot_game(test_dataset[0])
plot_in_basis(
    x,
    probes["+pee-ee"][..., 4],
    labels,
    n_cols=8,
    title="blocks.2.ln1.hook_normalized in PE-E basis",
)
plot_in_basis(
    x,
    b_padded,
    labels,
    n_cols=8,
    title="blocks.2.ln1.hook_normalized in B basis",
)

# %%
# plot_in_basis(
#     model.W_K[2, 5].T.detach().cpu(),
#     b_padded,
#     labels=list(range(1, 33)),
#     title="L2H5 K.b",
# )
d_head_labels = [f"d_head_{i}" for i in range(32)]
plot_in_basis(
    model.W_K[2, 5].T.detach(),
    probes["+pee-ee"][..., 4],
    labels=d_head_labels,
    title="L2H5 K.(pe-e)",
)
plot_in_basis(
    model.W_K[2, 5].T.detach(),
    p_padded,
    labels=d_head_labels,
    title="L2H5 K.p",
)
plot_in_basis(
    model.W_Q[2, 5].T.detach(),
    probes["+pee-ee"][..., 4],
    labels=d_head_labels,
    title="L2H5 Q.(pe-e)",
)
plot_in_basis(
    model.W_Q[2, 5].T.detach(),
    p_padded,
    labels=d_head_labels,
    title="L2H5 Q.p",
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
# Try to find a time where L2H5 doesn't attend to pos 0 or D5
# Either C2 at pos 1, C5 at pos 2, C2 at pos 2, B4 at pos 2
# Can see L2H5D3 will cancel D5 and boost pos 1 if dst has C2 at pos 1
d5_pos = (focus_input_ids == 23).argmax(-1)
for i in range(13, len(focus_batch)):
    a = focus_cache["blocks.2.attn.hook_pattern"][i, 5]
    a0 = a[:, 0]
    a_d5 = a[:, d5_pos[i]]
    max_attn = t.stack([a0, a_d5]).max(0)[0]
    if t.any(max_attn < 0.9):
        test_game = focus_batch[i]
        test_input_ids = t.tensor(test_game["input_ids"], device=device)
        if test_input_ids[1] == 8 or test_input_ids[2] == 22 or test_input_ids[2] == 8:
            continue
        test_logits, test_cache = model.run_with_cache(test_input_ids[:-1])
        vis = visualize_attention_patterns(
            list(range(model.cfg.n_layers * model.cfg.n_heads)),
            test_cache,
            test_game["moves"],
        )
        print(i)
        display(HTML(vis))
        plot_game(test_game)
        break

# %%
# Look for heads (or neurons) which attend to the ee_D5 output from L2H5
plot_in_basis(
    model.W_K[3, 0].T.detach(),
    probes["ee"]
)














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
