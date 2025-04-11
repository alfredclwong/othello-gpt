# %%
from pathlib import Path

import huggingface_hub as hf
import torch as t
import wandb
from datasets import load_dataset
from itertools import product

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
in_hook_suffix = "hook_resid_post"
out_hook_suffix = "ln_final.hook_normalized"
sae_cfg = SAEConfig(
    d_in=model.cfg.d_model,
    d_sae=1024,
    in_hook_layer=model.cfg.n_layers - 1,
    in_hook_suffix=in_hook_suffix,
    out_hook_layer=model.cfg.n_layers - 1,
    out_hook_suffix=out_hook_suffix,
    lr=1e-4,
    sparsity_coeff=0.001,
    n_epochs=4,
    resample_freq=9000,
    resample_window=2000,
    use_b_dec=True,
)
print(sae_cfg)
sae = SAE(sae_cfg, model, device)
sae.optimise(train_dataset, test_dataset.take(1024))
sae_name = f"{model_name}-sae-{sae.in_hook_name}"
if in_hook_suffix != out_hook_suffix:
    sae_name += f"-{sae.out_hook_name}"
sae.push_to_hub(sae_name)

# %%
hook_suffixes = [
    ("hook_resid_pre", "ln1.hook_normalized"),
    ("hook_resid_mid", "ln2.hook_normalized"),
]
sae_params = product(range(model.cfg.n_layers), hook_suffixes)
for i, (hook_layer, (in_hook_suffix, out_hook_suffix)) in enumerate(sae_params):
    sae_cfg = SAEConfig(
        d_in=model.cfg.d_model,
        d_sae=1024,
        in_hook_layer=hook_layer,
        in_hook_suffix=in_hook_suffix,
        out_hook_layer=hook_layer,
        out_hook_suffix=out_hook_suffix,
        lr=2e-3,
        sparsity_coeff=0.002,
        n_epochs=2,
        resample_freq=9000,
        resample_window=2000,
        use_b_dec=True,
    )
    print(sae_cfg)
    sae = SAE(sae_cfg, model, device)
    sae.optimise(train_dataset, test_dataset.take(1024))
    sae_name = f"{model_name}-sae-{sae.in_hook_name}"
    if in_hook_suffix != out_hook_suffix:
        sae_name += f"-{sae.out_hook_name}"
    sae.push_to_hub(sae_name)

# %%
prev_sparsity_coeffs = [0.05, 0.001, 0.02, 0.003, 0.02, 0.005]
sparsity_coeffs = [0.05, 0.001, 0.02, 0.005, 0.02, 0.005]
prev_lrs = [2e-3, 2e-3, 5e-3, 5e-3, 2e-2, 1e-2]
lrs = [2e-3, 2e-3, 5e-3, 5e-3, 2e-2, 5e-3]
hook_suffixes = [
    ("ln2.hook_normalized", "hook_mlp_out"),
    ("attn.hook_z", "attn.hook_z"),
]
sae_params = product(range(model.cfg.n_layers), hook_suffixes)
for i, (hook_layer, (in_hook_suffix, out_hook_suffix)) in enumerate(sae_params):
    is_mlp = "mlp" in out_hook_suffix
    last_mlp = is_mlp and (hook_layer == model.cfg.n_layers - 1)
    sae_cfg = SAEConfig(
        d_in=model.cfg.d_model,
        d_sae=2048 if last_mlp else 1024,
        in_hook_layer=hook_layer,
        in_hook_suffix=in_hook_suffix,
        out_hook_layer=hook_layer,
        out_hook_suffix=out_hook_suffix,
        lr=lrs[i],
        sparsity_coeff=sparsity_coeffs[i],
        n_epochs=4 if last_mlp else 2,
        resample_freq=9000,
        resample_window=2000,
        use_b_dec=not is_mlp,  # transcoders don't need a bias term in unbiased models

        # n_steps_per_epoch=1000,
        # log_steps=500,
        # use_wandb=False,
        # log_warmup_steps=100,
    )
    print(sae_cfg)
    sae = SAE(sae_cfg, model, device)
    sae.optimise(train_dataset, test_dataset.take(1024))
    sae_name = f"{model_name}-sae-{sae.in_hook_name}"
    if in_hook_suffix != out_hook_suffix:
        sae_name += f"-{sae.out_hook_name}"
    sae.push_to_hub(sae_name)

# %%
