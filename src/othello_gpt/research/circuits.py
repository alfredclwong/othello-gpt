# %%
import numpy as np
import torch as t
from datasets import load_dataset
import huggingface_hub as hf
from pathlib import Path
import einops
import plotly.graph_objects as go
from typing import Optional
from jaxtyping import Float
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer
from typing import List, Tuple

from othello_gpt.data.vis import plot_in_basis
from scipy.stats import kurtosis
from othello_gpt.util import (
    get_all_squares,
    load_model,
    load_probes,
)

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
model = load_model(device, "awonga/othello-gpt-7M")

# %%
induction_probes = ["+t-pt", "+m-pt", "+t-pm", "+m-pm", "+m-pe", "+e-pe"]
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
    combos=induction_probes,
)
probes = {k: probes[k] for k in list("temclu") + ["pt", "pe", "pm", *induction_probes]}
probe_layer = 50
{k: p.shape for k, p in probes.items()}  # d_model (row col) probe_layer

# %%
# Circuits:
#  1. TEM (spatial) -> L
#  2. PT,C' -> M-PT
#  3. PT,C  -> T-PT
#  4. PM,C' -> T-PM
#  5. PM,C  -> M-PM
#  i.e. [above] C -> T-M if PT, else M-T (prob needs non-linearity)

# %%
# Let's try to identify the same-square inductive circuits first (e.g. 2-5 above)
# We want neurons that align strongly across several probes at the same square
# Each neuron has w_in/w_out. We can get the alignment at each board position, filter by kurtosis,
# and classify according to the max
stacked_probes = t.stack(list(probes.values()))[..., probe_layer]  # shape: n_probe d_model (row col) [probe_layer]
w_all = t.stack([model.W_in, model.W_out.transpose(1, 2)])  # shape: io layer d_model d_mlp
alignments = einops.einsum(
  w_all,
  stacked_probes,
  "io layer d_model neuron, probe d_model n_out -> io layer neuron probe n_out"
).detach().cpu()
ekurts = kurtosis(alignments, axis=-1) - 3
ekurt_threshold = 3
filtered_indices = t.where(ekurts > ekurt_threshold)
filtered_alignments = alignments[..., filtered_indices[0], filtered_indices[1]]
# %%
max_abs_square = alignments.abs().argmax(-1)
modal_squares, _ = t.mode(max_abs_square, keepdim=True)
modal_squares

# max_indices = t.argmax(t.abs(filtered_alignments).sum(dim=0), dim=0)
# layer_neuron_max = {
#   (layer.item(), neuron.item()): max_indices[:, layer, neuron].tolist()
#   for layer in range(filtered_alignments.shape[2])
#   for neuron in range(filtered_alignments.shape[3])
# }
# layer_neuron_max

# %%
# Find neurons that write to ["+t-pt", "+m-pt", "+t-pm", "+m-pm", "+m-pe", "+e-pe"] in one specific square
# Plot their inputs in ptem+c
neurons = {}
w_out = model.W_out.detach()
induction_probes_stacked = t.stack([probes[k] for k in induction_probes])[..., probe_layer]
w_out_probed = einops.einsum(
  w_out,
  induction_probes_stacked,
  "layer neuron d_model, probe d_model n_out -> layer neuron probe n_out"
).cpu()
max_abs_square = w_out_probed.abs().argmax(-1)
modal_squares, _ = t.mode(max_abs_square, keepdim=True)
(max_abs_square == modal_squares).sum(-1)
# t.where(max_abs_square == modal_squares, w_out_probed, 0)

# kurts = kurtosis(w_out_probed, axis=-1)
# sorted_indices = np.argsort(-kurts)[:10]
# sorted_w_out_probed = w_out_probed[sorted_indices]
# print(k, w_out_probed[sorted_indices], max_abs_square[sorted_indices])

# %%
# Now let's try to identify circuit chains working backwards. At the last layer, the most important neurons
# are the ones that write out in the unembed direction. Find their conditional inputs and work backwards.