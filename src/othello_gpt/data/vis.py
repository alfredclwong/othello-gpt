from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def move_id_to_coord(move_id: int, size: int) -> tuple:
    return tuple(divmod(move_id, size))


def move_id_to_text(move_id: int, size: int) -> str:
    if move_id == -1:
        return "PAD"
    if move_id == size * size:
        return "PASS"
    y, x = move_id_to_coord(move_id, size)
    return f"{chr(ord('A') + x)}{y + 1}"


def plot_game(
    game: Dict[str, List],
    subplot_size=180,
    n_cols=8,
    reversed=True,
    textcolor=None,
    hovertext=None,
    shift_legalities=True,
    title="",
    subplot_titles=None,
):
    game_boards = np.array(game["boards"])
    n_moves, size, _ = game_boards.shape

    if "legalities" in game:
        game_legalities = np.array(game["legalities"])
    else:
        game_legalities = np.zeros_like(game_boards)

    if "moves" in game:
        game_moves = np.array(game["moves"])
    else:
        game_moves = np.full(game_boards.shape[0], size * size)

    row_labels = list(map(str, range(1, 1 + size)))
    col_labels = [chr(ord("A") + i) for i in range(size)]
    if hovertext is None:
        hovertext = np.array(
            [
                [[f"{col}{row}" for col in col_labels] for row in row_labels]
                for _ in range(n_moves)
            ]
        )
    margin = subplot_size // 8

    n_rows = (n_moves - 1) // n_cols + 1

    if subplot_titles is None:
        subplot_titles = [
            f"{i + 1}. {move_id_to_text(int(move_id), size)}"
            for i, move_id in enumerate(game_moves)
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=True,
        # vertical_spacing=0.1,
    )

    for i in range(n_moves):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Create str 2d array for legal moves where 0 -> "" and 1 -> "X"
        if i + int(shift_legalities) < n_moves:
            text = np.where(game_legalities[i + int(shift_legalities)], "X", "")
        else:
            text = np.full_like(game_legalities[0], "", dtype=str)
        if game_moves[i] != size * size:
            coord = move_id_to_coord(int(game_moves[i]), size)
            text[*coord] = "o"

        fig.add_trace(
            go.Heatmap(
                z=game_boards[i],
                colorscale="gray_r" if reversed else "gray",
                showscale=False,
                text=text,
                hovertext=hovertext[i],
                hovertemplate="%{hovertext}<extra></extra>",
                x=col_labels,
                y=row_labels,
                xgap=0.2,
                ygap=0.2,
                texttemplate="%{text}",
                textfont={
                    "color": textcolor if textcolor else "black" if i % 2 else "white"
                },
            ),
            row=row,
            col=col,
        )

    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        constrain="domain",
        autorange="reversed",
    )

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=subplot_size // 10)),
        title_x=0.5,
        font=dict(size=subplot_size // 20),
        margin=dict(l=margin, r=margin, t=margin * 3, b=margin),
        width=subplot_size * n_cols,
        height=subplot_size * n_rows,
    )

    fig.update_annotations(font_size=subplot_size // 10)

    fig.show()
