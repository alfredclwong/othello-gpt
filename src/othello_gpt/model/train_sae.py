# %%
from pathlib import Path

import huggingface_hub as hf
import torch as t
import wandb
from datasets import load_dataset

from othello_gpt.util import load_model
from othello_gpt.model.sae import OthelloSAE, OthelloSAEConfig

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

root_dir = Path().cwd().parent.parent.parent

hf.login((root_dir / "secret.txt").read_text())
wandb.login()

model_version = "300k"
model_name = f"awonga/othello-gpt-{model_version}"
model = load_model(device, model_name)

dataset_dict = load_dataset("awonga/othello-gpt")
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]

# %%
sae_cfg = OthelloSAEConfig(
    d_in=model.cfg.d_model,
    d_sae=2048,
    hook_layers=list(range(model.cfg.n_layers)),
    hook_suffixes=["attn.hook_z", "hook_mlp_out"],
    use_wandb=True,
    n_epochs=2,
    n_train=300*128,
    batch_size=128,
    resample_freq=100,
    resample_window=100,
    log_steps=100,
)
print(sae_cfg)
sae = OthelloSAE(sae_cfg, model, train_dataset, test_dataset, device)
data_log = sae.optimize()
sae.push_to_hub(f"{model_name}-sae")

# %%
