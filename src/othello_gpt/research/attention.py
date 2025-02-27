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

from othello_gpt.data.vis import plot_in_basis, plot_game
from othello_gpt.util import (
    get_all_squares,
    load_model,
    load_probes,
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
model = load_model(device, "awonga/othello-gpt-6M")

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
test_dataset = dataset_dict["test"].take(10)
test_game = test_dataset[0]
test_input_ids = t.tensor(test_game["input_ids"], device=device)
test_logits, test_cache = model.run_with_cache(test_input_ids[:-1])
vis = visualize_attention_patterns(
    list(range(model.cfg.n_layers * model.cfg.n_heads)),
    test_cache,
    test_game["moves"],
)
HTML(vis)

# %%
plot_game(test_game)

# %%
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    combos=["t+m", "t-m", "t-e", "t-pt", "m-pm"],
)
{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
(
    model.W_Q.shape,
    model.W_K.shape,
    model.W_V.shape,
    model.W_O.shape,
)  # n_layer n_head d_model d_head

# %%
n_layer = model.cfg.n_layers
n_head = model.cfg.n_heads
d_head = model.cfg.d_head
n_neuron = model.cfg.d_model * 4

# %%
w_ov = einops.einsum(
    model.W_O,
    model.W_V,
    "n_layer n_head d_head o, n_layer n_head v d_head -> n_layer n_head o v",
)
w_ov = einops.rearrange(
    w_ov, "n_layer n_head o v -> (n_layer n_head) o v"
).detach()
w_q = einops.rearrange(
    model.W_Q, "n_layer n_head d_model d_head -> (n_layer n_head d_head) d_model"
).detach()
w_k = einops.rearrange(
    model.W_K, "n_layer n_head d_model d_head -> (n_layer n_head d_head) d_model"
).detach()

w_ov /= w_ov.norm(dim=0, keepdim=True)
w_q /= w_q.norm(dim=0, keepdim=True)
w_k /= w_k.norm(dim=0, keepdim=True)

for layer in [2, 4]:
    for p in ["t", "e", "m", "ot", "om", "f", "l", "u"]:
        labels = [
            f"L{l} H{h} D{d}"
            for l in range(n_layer)
            for h in range(n_head)
            for d in range(d_head)
        ]

        plot_in_basis(
            t.randn_like(w_q),
            probes[p][..., layer],
            labels,
            top_n=5,
            title=f"R . {p}{layer}",
        )
        plot_in_basis(
            w_q,
            probes[p][..., layer],
            labels,
            top_n=5,
            title=f"W_Q . {p}{layer}",
        )
        plot_in_basis(
            w_k,
            probes[p][..., layer],
            labels,
            top_n=5,
            title=f"W_K . {p}{layer}",
        )

        labels = [
            f"L{l} H{h} D{d}"
            for l in range(n_layer)
            for h in range(n_head)
            for d in range(model.cfg.d_model)
        ]

        plot_in_basis(
            w_ov,
            probes[p][..., layer],
            labels,
            top_n=5,
            title=f"W_OV . {p}{layer}",
        )
