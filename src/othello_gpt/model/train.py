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
from eindex import eindex
import plotly.graph_objects as go

from othello_gpt.data.vis import plot_game, move_id_to_coord
from othello_gpt.model.nanoGPT import GPT, GPTConfig
from othello_gpt.util import pad_batch, get_all_squares, load_model

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
    n_layer=8,
    n_head=8,
    n_embd=256,
    dropout=0.0,
    bias=False,
    weight_tying=False,
)
print(cfg)
model = HubGPT(cfg).to(device)


# %%
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 512
    epochs: int = 32
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
model.push_to_hub("awonga/othello-gpt-6M")

# %%
all_squares = t.tensor(get_all_squares(size), device=device)
model = load_model(device, "awonga/othello-gpt-400k")
n_focus = 1000
focus_games = dataset_dict["test"].take(n_focus)
focus_input_ids = t.tensor(focus_games["input_ids"], device=device)
focus_logits = model(focus_input_ids[:, :-1])
focus_logit_boards = t.full((n_focus, focus_logits.shape[1], size, size), t.nan, device=device)
focus_logit_boards.flatten(2)[..., all_squares] = focus_logits
focus_prob_boards = t.full_like(focus_logit_boards, t.nan)
focus_prob_boards.flatten(2)[..., all_squares] = focus_logits.softmax(-1)
focus_preds = focus_logits.argmax(-1)
focus_pred_move_ids = all_squares[focus_preds]
focus_pred_boards = t.zeros((*focus_pred_move_ids.shape, size * size), device=device)
focus_pred_boards.scatter_(-1, focus_pred_move_ids.unsqueeze(-1), 1)
focus_pred_boards = focus_pred_boards.view(*focus_preds.shape, size, size)
focus_legalities = t.tensor(focus_games["legalities"], device=device)[:, 1:]

# %%
# Calculate % of predicted tokens (argmax) that are legal
next_token_legal_by_pos = eindex(
    focus_legalities.flatten(-2), focus_pred_move_ids, "batch pos [batch pos]"
)
next_token_legal_by_pos = next_token_legal_by_pos.float().mean(0).detach().cpu()

# Calculate % of distribution assigned to legal moves
legal_prob_by_pos = t.where(focus_legalities, focus_prob_boards, t.nan)
legal_prob_by_pos = legal_prob_by_pos.flatten(-2).nansum(-1).detach().cpu()

n_legal_by_pos = focus_legalities.float().flatten(-2).sum(-1).detach().cpu()

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=next_token_legal_by_pos,
    name='Probability that top-1 logit is legal',
    line=dict(color="blue"),
))

fig.add_trace(go.Scatter(
    y=legal_prob_by_pos.mean(0).detach().cpu(),
    name='Sum of probabilities assigned to legal moves',
    line=dict(color="green"),
))

fig.add_trace(go.Scatter(
    y=n_legal_by_pos.mean(0).detach().cpu(),
    name='Number of legal moves',
    yaxis='y2',
    line=dict(color="red"),
))

# Add dotted lines for averages
fig.add_trace(go.Scatter(
    y=next_token_legal_by_pos.mean().repeat(model.cfg.n_ctx).detach().cpu(),
    mode='lines',
    line=dict(dash='dot', color=fig.data[0].line.color),
    showlegend=False,
))

fig.add_trace(go.Scatter(
    y=legal_prob_by_pos.mean().repeat(model.cfg.n_ctx).detach().cpu(),
    mode='lines',
    line=dict(dash='dot', color=fig.data[1].line.color),
    showlegend=False,
))

fig.add_trace(go.Scatter(
    y=n_legal_by_pos.mean().repeat(model.cfg.n_ctx).detach().cpu(),
    mode='lines',
    line=dict(dash='dot', color=fig.data[2].line.color),
    showlegend=False,
    yaxis='y2'
))

fig.update_layout(
    yaxis2=dict(
        overlaying='y',
        side='right'
    )
)

fig.update_layout(
    title='Model Metrics by Position',
    xaxis_title='Position',
    yaxis_title='Metric Value',
    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5,font=dict(size=10),itemwidth=30),
)

fig.show()

# %%
test_index = 0
test_pred_model = {
    "boards": focus_prob_boards[test_index].nan_to_num(0).detach().cpu(),
    "legalities": focus_games[test_index]["legalities"][1:],
    "moves": focus_games[test_index]["moves"],
}

plot_game(focus_games[test_index], title="Ground truth board states and legal moves")
plot_game(
    test_pred_model,
    reversed=False,
    textcolor="red",
    hovertext=test_pred_model["boards"],
    title="Model predictions for legal moves",
    shift_legalities=False,
)

# %%
