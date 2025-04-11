from itertools import product
from transformer_lens import HookedTransformer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from othello_gpt.model.sae import SAE, SAEConfig


def load_saes(model: HookedTransformer, model_name: str, device: str) -> list[SAE]:
    # embed_cfg = SAEConfig(
    #     d_in=model.cfg.d_model,
    #     d_sae=1024,
    #     in_hook_layer=0,
    #     in_hook_suffix="ln1.hook_normalized",
    #     out_hook_layer=0,
    #     out_hook_suffix="ln1.hook_normalized",
    # )
    embed_ln1_cfg = SAEConfig(
        d_in=model.cfg.d_model,
        d_sae=1024,
        in_hook_layer=0,
        in_hook_suffix="hook_resid_pre",
        out_hook_layer=0,
        out_hook_suffix="ln1.hook_normalized",
    )
    attn_hook_suffixes = ("attn.hook_z", "attn.hook_z")
    mlp_hook_suffixes = ("ln2.hook_normalized", "hook_mlp_out")
    hook_suffixes = [attn_hook_suffixes, mlp_hook_suffixes]
    cfgs = [embed_ln1_cfg] + [
        SAEConfig(
            d_in=model.cfg.d_model,
            d_sae=2048 if i == 2 and "mlp" in out_hook_suffix else 1024,
            in_hook_layer=i,
            out_hook_layer=i,
            in_hook_suffix=in_hook_suffix,
            out_hook_suffix=out_hook_suffix,
        )
        for i, (in_hook_suffix, out_hook_suffix) in product(
            range(model.cfg.n_layers), hook_suffixes
        )
    ]
    hook_names = [
        f"blocks.{cfg.out_hook_layer}.{cfg.out_hook_suffix}"
        if cfg.in_hook_suffix == cfg.out_hook_suffix
        else f"blocks.{cfg.in_hook_layer}.{cfg.in_hook_suffix}-blocks.{cfg.out_hook_layer}.{cfg.out_hook_suffix}"
        for cfg in cfgs
    ]
    sae_names = [f"{model_name}-sae-{hook_name}" for hook_name in hook_names]
    saes: list[SAE] = [
        SAE.from_pretrained(
            sae_name,
            cfg=cfg,
            model=model,
            device=device,
        )
        for sae_name, cfg in zip(sae_names, cfgs)
    ]
    for sae in saes:
        sae.eval()
        sae.requires_grad_(False)

    return saes


def plot_evals(
    eval_dict: dict[str, list[float]],
    metrics: list[str],
    y_ranges: list[None | tuple[None | float, None | float]],
    saes: list[SAE],
    n_col: int = 2,
):
    model = saes[0].model
    n_row = int(np.ceil(len(metrics) / n_col))
    fig = make_subplots(
        rows=n_row,
        cols=n_col,
        subplot_titles=metrics,
    )
    fig.update_layout(
        showlegend=False,
        height=240 * n_row,
        width=240 * n_col,
        title_text="SAE Metrics",
        margin=dict(t=50, l=10, r=10, b=10),
    )

    for i, (metric, y_range) in enumerate(zip(metrics, y_ranges)):
        row, col = (x + 1 for x in divmod(i, n_col))

        if metric == "alive_pct":
            d_sae = np.array([sae.cfg.d_sae for sae in saes])
            n_dead = np.array([eval_dict["n_dead"][i] for i in range(len(saes))])
            y_data = 1 - n_dead / d_sae
            n_alive = d_sae - n_dead
            text_dict = dict(
                text=n_alive,
                textposition="inside",
                textfont=dict(color="white", size=12),
            )
        else:
            y_data = eval_dict[metric]
            text_dict = {}

        fig.add_trace(
            go.Bar(x=list(range(len(y_data))), y=y_data, name=metric, **text_dict),
            row=row,
            col=col,
        )
        fig.update_yaxes(range=y_range, row=row, col=col)
        if row == n_row:
            fig.update_xaxes(title_text="SAE Index", row=row, col=col)

        if metric == "x_norm":
            # Draw a labeled, dashed hline at y = d_model
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(y_data) - 0.5,
                y0=model.cfg.d_model,
                y1=model.cfg.d_model,
                line=dict(color="black", dash="dash"),
                row=row,
                col=col,
            )
            fig.add_annotation(
                x=-0.2,
                y=model.cfg.d_model,
                text="d_model",
                showarrow=False,
                font=dict(size=10, color="black"),
                xanchor="left",
                yanchor="bottom",
                row=row,
                col=col,
            )

    fig.show()
