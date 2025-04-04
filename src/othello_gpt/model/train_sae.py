# %%
from pathlib import Path

import huggingface_hub as hf
import torch as t
import wandb
from datasets import load_dataset

from othello_gpt.util import load_model
from othello_gpt.model.sae import SAE, SAEConfig

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
for hook_layer in range(model.cfg.n_layers):
    for in_hook_suffix, out_hook_suffix in [
        ("ln2.hook_normalized", "hook_mlp_out"),
        ("attn.hook_z", "attn.hook_z"),
    ]:
        is_mlp = "mlp" in out_hook_suffix
        last_mlp = is_mlp and (hook_layer == model.cfg.n_layers - 1)
        sparsity_coeff = 0.05 * pow(2, hook_layer) / (1 if is_mlp else 20)
        downstream_coeff = 0
        sae_cfg = SAEConfig(
            d_in=model.cfg.d_model,
            d_sae=1024,
            in_hook_layer=hook_layer,
            in_hook_suffix=in_hook_suffix,
            out_hook_layer=hook_layer,
            out_hook_suffix=out_hook_suffix,
            lr=1e-2 if last_mlp else 1e-3,
            sparsity_coeff=0.01 if last_mlp else sparsity_coeff,
            downstream_coeff=downstream_coeff,
            n_epochs=4 if last_mlp else 2,
            resample_freq=9000,
            resample_window=2000,
        )
        print(sae_cfg)
        sae = SAE(sae_cfg, model, device)
        sae.optimise(train_dataset, test_dataset.take(1024))
        sae_name = f"{model_name}-sae-{sae.in_hook_name}"
        if in_hook_suffix != out_hook_suffix:
            sae_name += f"-{sae.out_hook_name}"
        sae.push_to_hub(sae_name)

# %%
