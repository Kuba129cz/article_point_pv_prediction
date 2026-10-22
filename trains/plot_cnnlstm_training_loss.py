#!/usr/bin/env python3
"""Extract TensorBoard train/validation loss history for a CNN-LSTM run and plot it.

Expected TensorBoard scalar tags written by the training script:
    Loss/Train_scaled
    Loss/Validation_scaled

Example:
    PYTHONPATH=. python -u trains/plot_cnnlstm_training_loss.py
    PYTHONPATH=. python -u trains/plot_cnnlstm_training_loss.py --run_id cnnlstm_ensemble_seed123
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_run_dir(
    runs_dir: Path,
    run_id: str,
    explicit_run_dir: Optional[Path],
) -> Path:
    """Find the newest TensorBoard run folder matching run_id."""
    if explicit_run_dir is not None:
        if not explicit_run_dir.is_dir():
            raise FileNotFoundError(
                f"TensorBoard run directory does not exist: {explicit_run_dir}"
            )
        if not any(explicit_run_dir.rglob("events.out.tfevents.*")):
            raise FileNotFoundError(
                f"No TensorBoard event file found under: {explicit_run_dir}"
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
        available = sorted(
            path.name for path in runs_dir.iterdir()
            if path.is_dir() and any(path.rglob("events.out.tfevents.*"))
        )
        raise FileNotFoundError(
            f"No TensorBoard event directory found for run_id {run_id!r} under {runs_dir}.\n"
            f"Available event directories: {available}\n"
            "Check --run_id, --runs_dir, and whether SummaryWriter was enabled."
        )

    # If a seed was trained more than once, use the newest run directory.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_history(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = accumulator.Tags().get("scalars", [])
    if tag not in available_tags:
        raise KeyError(
            f"TensorBoard tag {tag!r} not found in {run_dir}. "
            f"Available scalar tags: {available_tags}"
        )
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def find_best_epoch(model_dir: Path, run_id: str) -> Optional[int]:
    """Read best_epoch from the training script's saved metrics JSON, if present."""
    path = model_dir / f"{run_id}_best_epoch_metrics.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        value = data.get("best_epoch")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_id",
        default="cnnlstm_ensemble_seed42",
        help="Exact run ID/prefix, e.g. cnnlstm_ensemble_seed42",
    )
    parser.add_argument(
        "--runs_dir", default=Path("runs"), type=Path,
        help="Directory containing TensorBoard run folders",
    )
    parser.add_argument(
        "--run_dir", default=None, type=Path,
        help="Optional exact TensorBoard run directory",
    )
    parser.add_argument(
        "--model_dir", default=Path("saved_models/cnnlstm"), type=Path,
        help="Model directory containing *_best_epoch_metrics.json",
    )
    parser.add_argument(
        "--output_dir", default=Path("figures/cnnlstm"), type=Path,
        help="Where to save the figure and CSV",
    )
    args = parser.parse_args()

    run_dir = find_run_dir(args.runs_dir, args.run_id, args.run_dir)
    train_steps, train_loss = scalar_history(run_dir, "Loss/Train_scaled")
    val_steps, val_loss = scalar_history(run_dir, "Loss/Validation_scaled")

    if not train_steps and not val_steps:
        raise ValueError(f"No training or validation loss values found in {run_dir}.")

    # TensorBoard steps are zero-based; display epochs as one-based.
    train_epochs = [step + 1 for step in train_steps]
    val_epochs = [step + 1 for step in val_steps]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / f"{args.run_id}_loss_curve"
    best_epoch = find_best_epoch(args.model_dir, args.run_id)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    if train_epochs:
        ax.plot(
            train_epochs, train_loss, marker="o", markersize=3.5,
            linewidth=1.8, label="Training loss",
        )
    if val_epochs:
        ax.plot(
            val_epochs, val_loss, marker="s", markersize=3.5,
            linewidth=1.8, label="Validation loss",
        )
    if best_epoch is not None:
        ax.axvline(
            best_epoch, linestyle="--", linewidth=1.2,
            label=f"Best epoch by validation Active MAE ({best_epoch})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss (standardized target space)")
    ax.set_title("CNN-LSTM training and validation loss")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)
    ax.set_xlim(left=1)

    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Export logged values for reuse or editing.
    train_df = pd.DataFrame({"epoch": train_epochs, "train_loss_scaled": train_loss})
    val_df = pd.DataFrame({"epoch": val_epochs, "validation_loss_scaled": val_loss})
    history_df = train_df.merge(val_df, on="epoch", how="outer").sort_values("epoch")
    csv_path = base.with_suffix(".csv")
    history_df.to_csv(csv_path, index=False)

    print(f"TensorBoard run: {run_dir}")
    print(f"Training epochs logged: {len(train_epochs)}")
    print(f"Validation epochs logged: {len(val_epochs)}")
    if best_epoch is not None:
        print(f"Best epoch (selected by validation Active MAE): {best_epoch}")
    else:
        print(f"No best-epoch JSON found at {args.model_dir / (args.run_id + '_best_epoch_metrics.json')}")
    print(f"Saved figure (PNG): {png_path}")
    print(f"Saved figure (PDF): {pdf_path}")
    print(f"Saved values (CSV): {csv_path}")


if __name__ == "__main__":
    main()
