"""MLflow wrapper: every run records its parameters, metrics, git commit and data provenance."""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"  # one local database, ignored by git
ARTIFACT_ROOT = ROOT / "mlartifacts"  # files attached to runs (figures, json)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _experiment_id(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=(ARTIFACT_ROOT / name).as_uri())


@contextmanager
def start_run(experiment: str, run_name: str, params: dict, data_dir: str | Path | None = None):
    """Open an MLflow run tagged with the code version; log params and the subset description."""
    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(experiment_id=_experiment_id(experiment), run_name=run_name) as run:
        mlflow.set_tags({
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": str(_git("status", "--porcelain", "--untracked-files=no") != ""),  # code differs from the commit
        })
        mlflow.log_params(params)
        if data_dir is not None:
            mlflow.log_artifact(str(Path(data_dir) / "subset_info.json"))
        yield run


def log_scores(scores: dict[str, float], prefix: str, step: int | None = None) -> None:
    """Log {'accuracy': 0.8} as 'val_accuracy' (prefix='val'), optionally at an epoch."""
    mlflow.log_metrics({f"{prefix}_{name}": value for name, value in scores.items()}, step=step)
