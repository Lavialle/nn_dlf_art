"""Read finished runs back from MLflow and draw report figures in one consistent style."""
from __future__ import annotations

import statistics

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from artguide.tracking import TRACKING_URI

# Validated categorical palette (colour-blind safe in this order) and recessive inks for text and grids
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def runs(experiment: str) -> pd.DataFrame:
    """Every finished run of an experiment, newest first (empty if the experiment does not exist)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        return pd.DataFrame()
    df = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'")
    return df.sort_values("start_time", ascending=False).reset_index(drop=True)


def latest_per_name(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The most recent run for each run name: a re-run replaces an older one, it is not averaged with it."""
    df = df[df["tags.mlflow.runName"].isin(names)]
    return df.drop_duplicates("tags.mlflow.runName", keep="first")


def history(run_id: str, metric: str) -> dict[int, float]:
    """{epoch: value} of a metric logged at every epoch."""
    return {m.step: m.value for m in MlflowClient(TRACKING_URI).get_metric_history(run_id, metric)}


def mean_std(values) -> tuple[float, float]:
    values = list(values)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def style(ax: plt.Axes) -> None:
    """Recessive frame: no top/right spines, light grid behind the data, muted tick labels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
