#!/usr/bin/env python3
"""Plot TensorBoard train/validation loss for an lstm_future training run.

The training script writes these scalar tags:
    Loss/Train_scaled
    Loss/Validation_scaled

Default run ID matches the first member trained by train_ensemble.sh:
    lstm_future_ensemble_seed42

Run from the project root, for example:
    PYTHONPATH=. python trains/plot_lstm_future_training_loss.py

Or select another seed:
    PYTHONPATH=. python trains/plot_lstm_future_training_loss.py \\
        --run_id lstm_future_ensemble_seed123
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_RUN_ID = "lstm_future_ensemble_seed42"


def find_run_dir(
    runs_dir: Path,
    run_id: str,
    explicit_run_dir: Optional[Path] = None,
) -> Path:
    """Find the TensorBoard folder for this run ID, choosing the newest if repeated."""
    if explicit_run_dir is not None:
        if not explicit_run_dir.is_dir():
            raise FileNotFoundError(
                f"TensorBoard run directory does not exist: {explicit_run_dir}"
            )
        if not any(explicit_run_dir.rglob("events.out.tfevents.*")):
            raise FileNotFoundError(
                f"No TensorBoard event files found under: {explicit_run_dir}"
            )
        return explicit_run_dir

    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    candidates = [
        path for path in runs_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(run_id)
        and any(path.rglob("events.out.tfevents.*"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No TensorBoard event directory found for run_id {run_id!r} under {runs_dir}. "
            "Check the run ID and that SummaryWriter wrote event files."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_history(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """Return TensorBoard step/value pairs for a scalar tag."""
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = accumulator.Tags().get("scalars", [])
    if tag not in available_tags:
        raise KeyError(
            f"TensorBoard tag {tag!r} not found in {run_dir}. "
            f"Available scalar tags: {available_tags}"
        )

    events = accumulator.Scalars(tag)
    if not events:
        raise ValueError(f"TensorBoard tag {tag!r} has no recorded values in {run_dir}.")
    return [event.step for event in events], [event.value for event in events]


def find_best_epoch(model_dir: Path, run_id: str) -> Optional[int]:
    """Read best epoch from the training script's metrics JSON, if available."""
    metrics_path = model_dir / f"{run_id}_best_epoch_metrics.json"
    if not metrics_path.is_file():
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        value = metrics.get("best_epoch")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_id",
        default=DEFAULT_RUN_ID,
        help="Run ID, e.g. lstm_future_ensemble_seed42",
    )
    parser.add_argument(
        "--runs_dir",
        default=Path("runs"),
        type=Path,
        help="Directory containing TensorBoard runs",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        type=Path,
        help="Optional exact TensorBoard run directory",
    )
    parser.add_argument(
        "--model_dir",
        default=Path("saved_models/lstm_future"),
        type=Path,
        help="Model directory containing <run_id>_best_epoch_metrics.json",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("figures/lstm_future"),
        type=Path,
        help="Directory for output figures and CSV",
    )
    args = parser.parse_args()

    run_dir = find_run_dir(args.runs_dir, args.run_id, args.run_dir)
    train_steps, train_loss = scalar_history(run_dir, "Loss/Train_scaled")
    val_steps, val_loss = scalar_history(run_dir, "Loss/Validation_scaled")

    # Training logs use zero-based epoch indices; display them from epoch 1.
    train_epochs = [step + 1 for step in train_steps]
    val_epochs = [step + 1 for step in val_steps]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / f"{args.run_id}_loss_curve"
    best_epoch = find_best_epoch(args.model_dir, args.run_id)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(
        train_epochs, train_loss,
        marker="o", markersize=3.5, linewidth=1.8,
        label="Training loss",
    )
    ax.plot(
        val_epochs, val_loss,
        marker="s", markersize=3.5, linewidth=1.8,
        label="Validation loss",
    )
    if best_epoch is not None:
        ax.axvline(
            best_epoch, linestyle="--", linewidth=1.2,
            label=f"Best epoch by validation Active MAE ({best_epoch})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss (standardized target space)")
    ax.set_title("LSTM Future: training and validation loss")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)
    ax.set_xlim(left=1)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    csv_path = output_base.with_suffix(".csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    train_df = pd.DataFrame({"epoch": train_epochs, "train_loss_scaled": train_loss})
    val_df = pd.DataFrame({"epoch": val_epochs, "validation_loss_scaled": val_loss})
    history_df = train_df.merge(val_df, on="epoch", how="outer").sort_values("epoch")
    history_df.to_csv(csv_path, index=False)

    print(f"TensorBoard run: {run_dir}")
    print(f"Training epochs logged: {len(train_epochs)}")
    print(f"Validation epochs logged: {len(val_epochs)}")
    if best_epoch is not None:
        print(f"Best epoch (selected by validation Active MAE): {best_epoch}")
    print(f"Saved figure (PNG): {png_path}")
    print(f"Saved figure (PDF): {pdf_path}")
    print(f"Saved values (CSV): {csv_path}")


if __name__ == "__main__":
    main()
