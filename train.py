import time
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from model_dinov2 import (
    DINOv2GazeClassifier,
    SessionDataset,
    _session_collate,
    _worker_init_fn,
    discover_sessions,
    evaluate_on_test,
)


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Train DINOv2 Gaze-Weighted Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--attn-mode",
        choices=["none", "gaze_bias", "fixation_mask"],
        default="none",
        help="How gaze modifies transformer attention.",
    )
    p.add_argument(
        "--pooling",
        choices=["gaze_weighted", "cls", "mean_patch"],
        default="gaze_weighted",
        help="How patch tokens are aggregated into a session embedding.",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the DINOv2 backbone (only train the classification head).",
    )

    p.add_argument("--epochs", type=int, default=100, help="Max training epochs per fold.")
    p.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds.")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the classification head.")
    p.add_argument("--backbone-lr", type=float, default=1e-6, help="Learning rate for the backbone (ignored if --freeze-backbone).")
    p.add_argument("--slices-per-step", type=int, default=8, help="CT slices sampled per training step (memory vs. coverage).")
    p.add_argument("--accumulate-grad", type=int, default=4, help="Gradient accumulation steps (simulates larger batch size).")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    p.add_argument("--checkpoint-dir", type=str, default=None, help="Where to save model checkpoints (default: ./checkpoints).")
    p.add_argument("--progress-bar", action="store_true", help="Show per-epoch progress bar.")

    return p.parse_args()


def train_one_fold(train_sessions, val_sessions, fold_idx, model_kwargs,
                   args, ckpt_dir):
    model = DINOv2GazeClassifier(**model_kwargs)

    train_ds = SessionDataset(
        train_sessions, model.transform, model.patch_grid,
        slices_per_step=args.slices_per_step,
    )
    val_ds = SessionDataset(
        val_sessions, model.transform, model.patch_grid,
        slices_per_step=None, # use all slices for validation
    )

    train_dl = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=_session_collate, num_workers=args.num_workers,
        worker_init_fn=_worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=_session_collate, num_workers=args.num_workers,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_auc", mode="max", save_top_k=1,
        filename=f"fold{fold_idx}-best",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[ckpt_cb],
        accumulate_grad_batches=args.accumulate_grad,
        enable_progress_bar=args.progress_bar,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_model = DINOv2GazeClassifier.load_from_checkpoint(ckpt_cb.best_model_path)
    best_auc = (ckpt_cb.best_model_score.item()
                if ckpt_cb.best_model_score is not None else 0.0)
    return best_model, best_auc


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    print("\n" + "=" * 50)
    print("DINOv2 Gaze-Weighted Classifier — Training")
    print("=" * 50)
    print(f"Attention mode: {args.attn_mode}")
    print(f"Pooling: {args.pooling}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Folds: {args.folds}")
    print(f"Head LR: {args.lr}")
    print(f"Backbone LR: {args.backbone_lr}")
    print(f"Slices/step: {args.slices_per_step}")
    print(f"Grad accumulation: {args.accumulate_grad}")
    print(f"Seed: {args.seed}")
    print("=" * 50)

    train_pool, test_set = discover_sessions()

    n_expert_train = sum(s["label"] == 1 for s in train_pool)
    n_novice_train = sum(s["label"] == 0 for s in train_pool)
    n_expert_test = sum(s["label"] == 1 for s in test_set)
    n_novice_test = sum(s["label"] == 0 for s in test_set)

    print(f"Train: {len(train_pool)} sessions ({n_expert_train} expert, {n_novice_train} novice)")
    print(f"Test:  {len(test_set)} sessions ({n_expert_test} expert, {n_novice_test} novice)")

    model_kwargs = dict(
        attn_mode=args.attn_mode,
        pooling=args.pooling,
        freeze_backbone=args.freeze_backbone,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
    )

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        ckpt_dir = str(Path(__file__).resolve().parent / "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoints: {ckpt_dir}")

    print(f"\n {args.folds}-Fold Cross-Validation")

    labels = [s["label"] for s in train_pool]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_models = []
    fold_val_aucs = []
    total_t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(range(len(train_pool)), labels)):
        train_sessions = [train_pool[i] for i in tr_idx]
        val_sessions = [train_pool[i] for i in va_idx]

        print(f"\n Fold {fold + 1}/{args.folds}  "
              f"(train={len(train_sessions)}, val={len(val_sessions)})")
        fold_t0 = time.time()

        model, val_auc = train_one_fold(
            train_sessions, val_sessions, fold,
            model_kwargs, args, ckpt_dir,
        )
        fold_models.append(model)
        fold_val_aucs.append(val_auc)

        print(f"Fold {fold + 1}/{args.folds}, val AUC = {val_auc:.4f}  "
              f"({time.time() - fold_t0:.0f}s)")

    mean_auc = np.mean(fold_val_aucs)
    std_auc = np.std(fold_val_aucs)
    print(f"\n Mean val AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    print("\n Test Set Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, threshold, metrics = evaluate_on_test(fold_models, test_set, device)

    print(f"Threshold (Youden's J): {threshold:.3f}")
    print(f"ROC AUC: {metrics['ROC_AUC']:.4f}")
    print(f"F1: {metrics['F1']:.4f}")
    print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"Specificity: {metrics['Specificity']:.4f}")

    total_time = time.time() - total_t0
    print(f"\nTotal training time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
