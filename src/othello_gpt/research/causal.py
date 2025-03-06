# %%
from itertools import product
from pathlib import Path

import numpy as np
import einops
import pandas as pd
import plotly.graph_objects as go
import torch as t
import wandb
from datasets import load_dataset
from jaxtyping import Float
from plotly.subplots import make_subplots
from tqdm import tqdm
from sklearn.manifold import TSNE

from othello_gpt.data.vis import (
    plot_game,
    plot_probe_preds,
    plot_in_basis,
    move_id_to_text,
)
from othello_gpt.research.targets import (
    theirs_empty_mine_target,
    captures_target,
    legality_target,
    prev_tem_target,
    tm_target,
    l_if_e_target,
    ptm_target,
    empty_target,
    prev_empty_target,
    t_npt_target,
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
actually_all_squares = list(range(size*size))
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
# version = "1.5M"
version = "6M"
model = load_model(device, f"awonga/othello-gpt-{version}")
model

# %%
padded_W_pos = t.full((size * size, model.W_pos.shape[1]), t.nan, device=device)
padded_W_pos[:model.W_pos.shape[0], :] = model.W_pos
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
{k: p.shape for k, p in probes.items()}  # d_model (row col) n_probe_layer

# %%

