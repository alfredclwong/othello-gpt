# %%
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub as hf
import numpy as np
import torch as t
import wandb
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from othello_gpt.data.vis import plot_game
from othello_gpt.model.nanoGPT import GPT, GPTConfig
from othello_gpt.util import pad_batch, get_all_squares

# %%
device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)
device

# %%
root_dir = Path().cwd().parent.parent.parent
data_dir = root_dir / "data"
size = 6

hf.login(token=(root_dir / "secret.txt").read_text())
wandb.login()

# %%
dataset_dict = load_dataset("awonga/othello-gpt")
# plot_game(dataset_dict["test"][0], subplot_size=180, n_cols=8)


# %%
class HubGPT(GPT, hf.PyTorchModelHubMixin):
    pass


cfg = GPTConfig(
    block_size=(size * size - 4) - 1,
    vocab_size=size * size - 4,  # no pad
    n_layer=30,
    n_head=8,
    n_embd=36 * 4,
    dropout=0.0,
    bias=True,
)
print(cfg)
model = HubGPT(cfg).to(device)


# %%
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 256
    epochs: int = 16
    max_steps_per_epoch: int = 1000
    lr: int = 1e-3
    weight_decay: int = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)
    wandb_project: str | None = "othello-gpt"
    wandb_name: str | None = None


args = TransformerTrainingArgs()


# %%
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: GPT):
        super().__init__()
        self.model = model
        self.args = args

        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=args.betas,
        )
        self.step = 0

        def collate_fn(batch):
            return pad_batch(batch, model.config.block_size + 1)

        self.train_loader = DataLoader(
            dataset_dict["train"]["input_ids"],
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"]["input_ids"],
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def training_step(self, batch: Int[Tensor, "batch seq"]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        _, loss = model(batch[:, :-1], batch[:, 1:])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        total_correct, total_samples = 0, 0

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            batch = batch.to(device)
            logits, _ = self.model(batch[:, :-1], batch[:, 1:])
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == batch[:, 1:]).sum().item()
            total_samples += batch.size(0) * (batch.size(1) - 1)

        accuracy = total_correct / total_samples
        wandb.log({"accuracy": accuracy}, step=self.step)
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        config_dict = model.config.__dict__.copy()
        config_dict.update(args.__dict__)
        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=config_dict,
        )
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch.to(device))
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()

        wandb.finish()


trainer = TransformerTrainer(args, model)
trainer.train()

# %%
model.push_to_hub("awonga/othello-gpt-7M")

# %%
n_focus = 50
focus_games = dataset_dict["test"].take(n_focus)
focus_input_ids = pad_batch(focus_games["input_ids"], max_len=cfg.block_size + 1).to(
    device
)
focus_logits, loss = model(focus_input_ids[:, :-1], focus_input_ids[:, 1:])
focus_logit_boards = t.full((n_focus, focus_logits.shape[1], size, size), 0.0)
focus_logit_boards.flatten(2)[..., get_all_squares(size)] = focus_logits.detach().cpu()

# %%
test_index = 0
test_pred_model = {
    "boards": focus_logit_boards[test_index].detach().cpu(),
    "legalities": focus_games[test_index]["legalities"],
    "moves": focus_games[test_index]["moves"],
}

plot_game(focus_games[test_index], title="Ground truth board states and legal moves")
plot_game(
    test_pred_model,
    reversed=False,
    textcolor="red",
    hovertext=test_pred_model["boards"],
    title="Model predictions for legal moves",
)

# %%
