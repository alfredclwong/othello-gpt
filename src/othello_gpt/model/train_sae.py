# %%
from pathlib import Path

import huggingface_hub as hf
import torch as t
import wandb
from datasets import load_dataset

from othello_gpt.util import load_model
from othello_gpt.model.sae import DownstreamSAE, DownstreamSAEConfig

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

model_version = "600k"
model_name = f"awonga/othello-gpt-{model_version}"
model = load_model(device, model_name)

dataset_dict = load_dataset("awonga/othello-gpt")
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]

# %%
for hook_layer in range(model.cfg.n_layers)[::-1]:
    for hook_suffix in ["attn.hook_z", "hook_mlp_out"][::-1]:
        is_mlp = "mlp" in hook_suffix
        sparsity_coeff = 0.05 * pow(2, hook_layer) / (1 if is_mlp else 20)
        # downstream_coeff = sparsity_coeff
        downstream_coeff = 0
        sae_cfg = DownstreamSAEConfig(
            d_in=model.cfg.d_model,
            d_sae=1024,
            hook_layer=hook_layer,
            hook_suffix=hook_suffix,
            lr=1e-3,
            sparsity_coeff=sparsity_coeff,
            downstream_coeff=downstream_coeff,
            n_epochs=2,
            resample_freq=9000,
            resample_window=2000,
        )
        print(sae_cfg)
        sae = DownstreamSAE(sae_cfg, model, device)
        sae.optimise(train_dataset, test_dataset.take(1024))
        sae.push_to_hub(f"{model_name}-sae-{sae.hook_name}")

# %%
