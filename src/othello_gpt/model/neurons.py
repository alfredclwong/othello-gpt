# %%
import torch as t
from datasets import load_dataset
import huggingface_hub as hf
from pathlib import Path
import einops
import numpy as np
from transformer_lens import HookedTransformerConfig, HookedTransformer
import plotly.graph_objects as go
from typing import Union, List, Optional
from jaxtyping import Float
from transformer_lens import ActivationCache
import circuitsvis as cv
from IPython.display import HTML

from othello_gpt.data.vis import plot_game
from othello_gpt.model.nanoGPT import GPT, GPTConfig
from scipy.stats import kurtosis
from othello_gpt.util import (
    convert_nanogpt_to_transformer_lens_weights,
    get_all_squares,
    pad_batch,
    get_id_to_token_id_map,
)
from othello_gpt.data.vis import move_id_to_text

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
probe_dir = data_dir / "probes"

hf.login((root_dir / "secret.txt").read_text())
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

linear_probe = t.load(probe_dir / "linear_probe_20250206_194246.pt", weights_only=True)

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
n_layer = 2
n_neuron = 512
w_out = model.W_out[:, :n_neuron]
w_out /= w_out.norm(dim=-1, keepdim=True)
w_in = model.W_in[:, :n_neuron].transpose(1, 2)
w_in /= w_in.norm(dim=-1, keepdim=True)
w_u = model.W_U[:, 1:].clone()
w_u /= w_u.norm(dim=0, keepdim=True)
probe_normed = linear_probe[..., 1].clone()
probe_normed /= probe_normed.norm(dim=0, keepdim=True)
print(probe_normed.shape)

# %%
neuron_unembed_board = t.full((n_layer, n_neuron, size, size), 0.0, device=device)
neuron_unembed_board.flatten(-2)[..., all_squares] = einops.einsum(
    w_out, w_u,
    "n_layer n_neuron d_model, d_model n_vocab -> n_layer n_neuron n_vocab"
)
neuron_unembed_board = neuron_unembed_board.flatten(0, 1).detach().cpu()
labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]

# sort by kurtosis (should find sparse neurons)
kurt_vals = kurtosis(neuron_unembed_board.numpy(), axis=(1, 2), fisher=False)
sorted_indices = np.argsort(-kurt_vals)[:128]

# sort by abs value of A1 square
# a1_abs = neuron_unembed_board[:, 0, 0].abs()
# sorted_indices = t.argsort(-a1_abs)[:128]

neuron_unembed_board = neuron_unembed_board[sorted_indices]
labels = [labels[i] for i in sorted_indices]

plot_game(
    {"boards": neuron_unembed_board},
    n_cols=5,
    hovertext=neuron_unembed_board,
    reversed=False,
    subplot_titles=labels,
)

# %%

w_out_probed_board = einops.einsum(
    w_out,
    probe_normed[..., 0],  # 0 = theirs, 1 = empty, 2 = mine
    "layer n_neuron d_model, d_model row col -> layer n_neuron row col"
)
w_out_probed_board = w_out_probed_board.flatten(0, 1).detach().cpu()

labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]

# sort by kurtosis (should find sparse neurons)
kurt_vals = kurtosis(w_out_probed_board.numpy(), axis=(1, 2), fisher=False)
sorted_indices = np.argsort(-kurt_vals)[:50]

w_out_probed_board = w_out_probed_board[sorted_indices]
labels = [labels[i] for i in sorted_indices]

plot_game(
    {"boards": w_out_probed_board},
    n_cols=5,
    hovertext=w_out_probed_board,
    reversed=False,
    subplot_titles=labels,
)

# %%
w_in_probed_board = einops.einsum(
    w_in,
    probe_normed[..., 0],  # 0 = theirs, 1 = empty, 2 = mine
    "layer n_neuron d_model, d_model row col -> layer n_neuron row col"
)
w_in_probed_board = w_in_probed_board.flatten(0, 1).detach().cpu()

labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]

# sort by kurtosis (should find sparse neurons)
kurt_vals = kurtosis(w_in_probed_board.numpy(), axis=(1, 2), fisher=False)
sorted_indices = np.argsort(-kurt_vals)[:50]

w_in_probed_board = w_in_probed_board[sorted_indices]
labels = [labels[i] for i in sorted_indices]

plot_game(
    {"boards": w_in_probed_board},
    n_cols=5,
    hovertext=w_in_probed_board,
    reversed=False,
    subplot_titles=labels,
    title="W_in x 'their' probe"
)

# %%
w_in_probed_board = einops.einsum(
    w_in,
    probe_normed[..., 1],  # 0 = theirs, 1 = empty, 2 = mine
    "layer n_neuron d_model, d_model row col -> layer n_neuron row col"
)
w_in_probed_board = w_in_probed_board.flatten(0, 1).detach().cpu()

labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]

# sort by kurtosis (should find sparse neurons)
kurt_vals = kurtosis(w_in_probed_board.numpy(), axis=(1, 2), fisher=False)
sorted_indices = np.argsort(-kurt_vals)[:50]

w_in_probed_board = w_in_probed_board[sorted_indices]
labels = [labels[i] for i in sorted_indices]

plot_game(
    {"boards": w_in_probed_board},
    n_cols=5,
    hovertext=w_in_probed_board,
    reversed=False,
    subplot_titles=labels,
    title="W_in x 'empty' probe",
)

# %%
w_in_probed_board = einops.einsum(
    w_in,
    probe_normed[..., 2],  # 0 = theirs, 1 = empty, 2 = mine
    "layer n_neuron d_model, d_model row col -> layer n_neuron row col"
)
w_in_probed_board = w_in_probed_board.flatten(0, 1).detach().cpu()

labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]

# sort by kurtosis (should find sparse neurons)
kurt_vals = kurtosis(w_in_probed_board.numpy(), axis=(1, 2), fisher=False)
sorted_indices = np.argsort(-kurt_vals)[:50]

w_in_probed_board = w_in_probed_board[sorted_indices]
labels = [labels[i] for i in sorted_indices]

plot_game(
    {"boards": w_in_probed_board},
    n_cols=5,
    hovertext=w_in_probed_board,
    reversed=False,
    subplot_titles=labels,
    title="W_in x 'mine' probe",
)

# %%
w_in_unembed_board = t.full((n_layer, n_neuron, size, size), 0.0, device=device)
w_in_unembed_board.flatten(-2)[..., all_squares] = einops.einsum(
    w_in, w_u,
    "n_layer n_neuron d_model, d_model n_vocab -> n_layer n_neuron n_vocab"
)
w_in_unembed_board = w_in_unembed_board.flatten(0, 1).detach().cpu()
labels = [f"L{l} N{n}" for l in range(n_layer) for n in range(n_neuron)]

# sort by kurtosis (should find sparse neurons)
kurt_vals = kurtosis(w_in_unembed_board.numpy(), axis=(1, 2), fisher=False)
sorted_indices = np.argsort(-kurt_vals)[:50]

w_in_unembed_board = w_in_unembed_board[sorted_indices]
labels = [labels[i] for i in sorted_indices]

plot_game(
    {"boards": w_in_unembed_board},
    n_cols=5,
    hovertext=w_in_unembed_board,
    reversed=False,
    subplot_titles=labels,
)

# %%
# Conditional on A1 being strongly (un)predicted, which L1 neurons activated strongly?
# Which post-L0 residual stream directions activated the L1 neurons?
# How did the L0 block create these directions?
cosine_similarity = t.nn.functional.cosine_similarity(
    probe_normed[..., 0].flatten(), probe_normed[..., 2].flatten(), dim=0
)
print("Cosine similarity:", cosine_similarity.item())

# %%
model.W_Q.shape, model.W_K.shape, model.W_V.shape, model.W_O.shape  # n_layer n_head d_model d_head

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
    )

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = cv.circuitsvis.attention.attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"

# %%
test_dataset = dataset_dict["test"].take(1000)
test_game = test_dataset[0]
test_input_ids = pad_batch([test_game["input_ids"]], max_len=model.cfg.n_ctx + 1)
test_logits, test_cache = model.run_with_cache(test_input_ids[:, :-1])
vis = visualize_attention_patterns(
    list(range(4)),
    test_cache,
    test_game["moves"],
)
HTML(vis)

# %%
w_o = model.W_O
w_v = model.W_V
w_ov = einops.einsum(
    w_o, w_v,
    "n_layer n_head d_head d_model_0, n_layer n_head d_model_1 d_head -> n_layer n_head d_model_0 d_model_1"
)
eigenvalues, _ = t.linalg.eig(w_ov.flatten(0, 1).detach().cpu())
eigenvalues /= eigenvalues.abs()


import plotly.express as px

# Prepare data for scatter plot
i = 3
eigenvalues_real = eigenvalues[i].real.numpy()
eigenvalues_imag = eigenvalues[i].imag.numpy()

# Create scatter plot using Plotly
fig = px.scatter(
    x=eigenvalues_real,
    y=eigenvalues_imag,
    labels={"x": "Real part", "y": "Imaginary part"},
    title="Scatter plot of eigenvalues"
)
fig.update_layout(xaxis_title="Real part", yaxis_title="Imaginary part")
fig.show()

# %%
