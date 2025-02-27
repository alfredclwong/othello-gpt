# %%
import itertools
from itertools import product
from pathlib import Path
from typing import List

import graphviz
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

from othello_gpt.data.vis import plot_game, plot_probe_preds, move_id_to_text
from othello_gpt.research.targets import (
    captures_target,
    flip_dir_target,
    legality_target,
    next_move_target,
    prev_tem_target,
    theirs_empty_mine_target,
    tm_target,
    l_if_e_target,
)
from othello_gpt.util import get_all_squares, load_model, load_probes, test_linear_probe

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
n_test = 100
test_dataset = dataset_dict["test"].take(n_test)

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

# %%
model = load_model(device, "awonga/othello-gpt-4M")
model

# %%
probes = load_probes(
    probe_dir,
    device,
    w_u=model.W_U.detach(),
    w_e=model.W_E.T.detach(),
)
{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%
input_ids = t.tensor(test_dataset["input_ids"], device=device)
_, cache = model.run_with_cache(
    input_ids[:, :-1]
)
X, labels = cache.get_full_resid_decomposition(
    apply_ln=True, return_labels=True, expand_neurons=True
)

# %%
# I want to represent a model as a computational graph
# The level of granularity is attention heads and neurons
# It should be a DAG for forward passes
dot = graphviz.Digraph("model")
dot.node("L0H0")
dot.node("L0H1")
dot.node()

# class ComputationalGraph:
#     nodes: List
#     edges: List
