import os
import json
import argparse
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter


MODEL_NAME = 'vit_base_patch14_dinov2'
EMBED_DIM = 768
SLICES_PER_STEP = 8         
INFERENCE_BATCH_SIZE = 32   
ACCUMULATE_GRAD_BATCHES = 4 
WINDOW_CENTER = -600
WINDOW_WIDTH = 1500
LABEL_MAPPING = {'A': 1, 'B': 1, 'C': 1, 'D': 0, 'E': 0}
TEST_RADS = {'C', 'E'}
TRAIN_POOL_RADS = {'A', 'B', 'D'}

BASE_PATHS = [
    'output_radA_combined',
    'output_radB_combined',
    'output_radC_combined',
    'output_radD_combined',
    'output_radE_combined',
]

SKIP_FOLDERS = {
    "..",
}

HEATMAP_FILE = 'heatmaps_1vis_degree_full.npy'
METADATA_FILE = 'metadata_1deg.json'

N_FOLDS = 5
EPOCHS = 100
LR = 1e-5
BACKBONE_LR = 1e-6
HIDDEN_DIM = 256
NUM_CLASSES = 2
SEED = 42
GAZE_ATTN_INIT_SCALE = 1.0   
FIXATION_THRESHOLD = 0.0 


def load_ct_volume(nifti_path):
    ct_data = nib.load(nifti_path).get_fdata()
    return np.flip(np.rot90(ct_data, k=-1, axes=(0, 1)), axis=1)


def window_ct(ct_slice,
              wl=WINDOW_CENTER,
              ww=WINDOW_WIDTH):
    lo, hi = wl - ww / 2, wl + ww / 2
    out = np.clip(ct_slice, lo, hi)
    return ((out - lo) / (hi - lo) * 255).astype(np.uint8)

def _heatmap_to_patch_weights(heatmap,
                              patch_grid,
                              n_patches):
    h = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(h, (patch_grid, patch_grid))
    w = pooled.squeeze().flatten().numpy()
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.ones(n_patches, dtype=np.float32) / n_patches
    return w.astype(np.float32)


class SessionDataset(Dataset):
    # Each item = one eye-tracking session.
    # Returns: images (K, 3, H, W), gaze_weights (K, N_PATCHES), label.

    def __init__(self, session_infos, transform, patch_grid,
                 slices_per_step=None):
        self.sessions = session_infos
        self.transform = transform
        self.patch_grid = patch_grid
        self.n_patches = patch_grid ** 2
        self.slices_per_step = slices_per_step

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        info = self.sessions[idx]

        volume = load_ct_volume(info['nifti_path']) # (H, W, D)
        heatmaps = np.load(info['heatmap_path']) # (D, 512, 512)
        n_slices = volume.shape[2]

        if self.slices_per_step and self.slices_per_step < n_slices:
            indices = np.sort(
                np.random.choice(n_slices, self.slices_per_step, replace=False)
            )
        else:
            indices = np.arange(n_slices)

        images, weights = [], []
        for i in indices:
            ct_slice = window_ct(volume[:, :, i])
            rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)
            images.append(self.transform(Image.fromarray(rgb)))
            weights.append(
                _heatmap_to_patch_weights(
                    heatmaps[i], self.patch_grid, self.n_patches,
                )
            )

        return (
            torch.stack(images),
            torch.from_numpy(np.stack(weights)), 
            torch.tensor(info['label'], dtype=torch.long),
        )


def _session_collate(batch):
    return batch[0]

def _worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class DINOv2GazeClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name=MODEL_NAME,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        lr=LR,
        backbone_lr=BACKBONE_LR,
        freeze_backbone=False,
        pooling='gaze_weighted',
        attn_mode='none',
        fixation_threshold=FIXATION_THRESHOLD,
        gaze_attn_init_scale=GAZE_ATTN_INIT_SCALE,
        inference_batch_size=INFERENCE_BATCH_SIZE,):

        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0,
        )
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        cfg = resolve_data_config(self.backbone.pretrained_cfg)
        self._transform = create_transform(**cfg, is_training=False)
        img_size = cfg['input_size'][-1]
        patch_size = self.backbone.patch_embed.patch_size[0]
        self.patch_grid = img_size // patch_size
        self.n_patches = self.patch_grid ** 2

        if attn_mode == 'gaze_bias':
            self.gaze_attn_scale = nn.Parameter(
                torch.tensor(gaze_attn_init_scale)
            )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

        self._val_probs: list = []
        self._val_labels: list = []

    @property
    def transform(self):
        return self._transform

    def _gaze_to_attn_bias(self, gaze_weights):
        B = gaze_weights.shape[0]
        bias = self.gaze_attn_scale * torch.log(gaze_weights + 1e-8)
        cls_bias = torch.zeros(B, 1, device=bias.device, dtype=bias.dtype)
        bias = torch.cat([cls_bias, bias], dim=1) # (B, 1+N)
        return bias[:, None, None, :] # (B, 1, 1, 1+N)

    def _fixation_to_attn_mask(self, gaze_weights):
        B = gaze_weights.shape[0]
        mask = gaze_weights > self.hparams.fixation_threshold  # (B, N_PATCHES)
        cls_vis = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_vis, mask], dim=1) # (B, 1+N)
        return mask[:, None, None, :] # (B, 1, 1, 1+N)

    def _backbone_forward_with_gaze(self,
                                    images,
                                    gaze_weights):
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)

        if self.hparams.attn_mode == 'gaze_bias':
            attn_mod = self._gaze_to_attn_bias(gaze_weights)
        else: # fixation_mask
            attn_mod = self._fixation_to_attn_mask(gaze_weights)

        for block in self.backbone.blocks:
            x = self._block_forward(block, x, attn_mod)

        x = self.backbone.norm(x)
        return x

    def _block_forward(self, block, x, attn_mod):
        x = x + block.drop_path1(block.ls1(
            self._attn_forward(block.attn, block.norm1(x), attn_mod)
        ))
        x = x + block.drop_path2(block.ls2(
            block.mlp(block.norm2(x))
        ))
        return x

    def _attn_forward(self, attn_module, x, attn_mod):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, attn_module.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # each (B, H, N, head_dim)

        if hasattr(attn_module, 'fused_attn') and attn_module.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mod,
                dropout_p=attn_module.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * attn_module.scale
            attn_weights = q @ k.transpose(-2, -1) # (B, H, N, N)
            if attn_mod.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attn_mod, float('-inf'))
            else:
                attn_weights = attn_weights + attn_mod
            attn_weights = attn_weights.softmax(dim=-1)
            attn_weights = attn_module.attn_drop(attn_weights)
            x = attn_weights @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    def forward(self, images, gaze_weights):
        K = images.shape[0]
        bs = self.hparams.inference_batch_size
        pooling = self.hparams.pooling
        attn_mode = self.hparams.attn_mode
        all_embs = []

        for start in range(0, K, bs):
            end = min(start + bs, K)
            img_batch = images[start:end]
            gw_batch = gaze_weights[start:end]

            if attn_mode != 'none':
                tokens = self._backbone_forward_with_gaze(img_batch, gw_batch)
            else:
                tokens = self.backbone.forward_features(img_batch)

            cls_tok = tokens[:, 0, :] # (b, D)
            patch_tok = tokens[:, 1:, :] # (b, N, D)

            if pooling == 'gaze_weighted':
                emb = torch.einsum('bp,bpd->bd', gw_batch, patch_tok)
            elif pooling == 'cls':
                emb = cls_tok
            else:  # mean_patch
                emb = patch_tok.mean(dim=1)

            all_embs.append(emb)

        slice_embs = torch.cat(all_embs, dim=0) # (K, D)
        session_emb = slice_embs.mean(dim=0, keepdim=True) # (1, D)
        return self.head(session_emb) # (1, C)

    def on_train_epoch_start(self):
        if self.hparams.freeze_backbone:
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        images, gaze_weights, label = batch
        logits = self(images, gaze_weights)            # (1, C)
        loss = self.criterion(logits, label.unsqueeze(0))
        self.log('train_loss', loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gaze_weights, label = batch
        logits = self(images, gaze_weights)
        loss = self.criterion(logits, label.unsqueeze(0))
        self.log('val_loss', loss, prog_bar=True, batch_size=1)
        probs = torch.softmax(logits, dim=-1)
        self._val_probs.append(probs[0, 1].detach().cpu())
        self._val_labels.append(label.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_probs:
            return
        all_probs = torch.stack(self._val_probs).numpy()
        all_labels = torch.stack(self._val_labels).numpy()
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        self.log('val_auc', auc, prog_bar=True, batch_size=1)
        self._val_probs.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        head_params = list(self.head.parameters())
        if hasattr(self, 'gaze_attn_scale'):
            head_params.append(self.gaze_attn_scale)

        if self.hparams.freeze_backbone:
            return torch.optim.Adam(head_params, lr=self.hparams.lr)

        return torch.optim.Adam([
            {'params': self.backbone.parameters(),
             'lr': self.hparams.backbone_lr},
            {'params': head_params,
             'lr': self.hparams.lr},
        ])


def discover_sessions():
    train_pool, test_set = [], []

    for bp in BASE_PATHS:
        if not os.path.isdir(bp):
            print(f"Base path missing: {bp}")
            continue
        for folder in sorted(os.listdir(bp)):
            if folder in SKIP_FOLDERS:
                continue
            session_dir = os.path.join(bp, folder)
            if not os.path.isdir(session_dir):
                continue

            meta_path = os.path.join(session_dir, 'metadata.json')
            our_dir = os.path.join(session_dir, 'miccai2026', 'our')
            our_meta_path = os.path.join(our_dir, METADATA_FILE)
            heatmap_path = os.path.join(our_dir, HEATMAP_FILE)

            if not all(os.path.isfile(p)
                       for p in [meta_path, our_meta_path, heatmap_path]):
                continue

            with open(meta_path) as f:
                meta = json.load(f)
            with open(our_meta_path) as f:
                our_meta = json.load(f)

            rad = meta.get('rad')
            label = LABEL_MAPPING.get(rad)
            if label is None:
                continue

            nifti_path = our_meta.get('nifti_path', '')
            if not os.path.isfile(nifti_path):
                continue

            info = {
                'nifti_path': nifti_path,
                'heatmap_path': heatmap_path,
                'label': label,
                'rad': rad,
                'ct_id': meta.get('CT_ID', ''),
                'folder': folder,
            }
            if rad in TEST_RADS:
                test_set.append(info)
            elif rad in TRAIN_POOL_RADS:
                train_pool.append(info)

    return train_pool, test_set

def train_one_fold(train_sessions, val_sessions, fold_idx, model_kwargs,
                   slices_per_step=SLICES_PER_STEP):
    model = DINOv2GazeClassifier(**model_kwargs)

    train_ds = SessionDataset(
        train_sessions, model.transform, model.patch_grid,
        slices_per_step=slices_per_step,
    )
    val_ds = SessionDataset(
        val_sessions, model.transform, model.patch_grid,
        slices_per_step=None, # all slices
    )

    train_dl = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=_session_collate, num_workers=2,
        worker_init_fn=_worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=_session_collate, num_workers=2,
    )

    ckpt_cb = ModelCheckpoint(
        monitor='val_auc', mode='max', save_top_k=1,
        filename=f'fold{fold_idx}-best',
    )
    print('CUDA:', torch.cuda.is_available())
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[ckpt_cb],
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_model = DINOv2GazeClassifier.load_from_checkpoint(
        ckpt_cb.best_model_path,
    )
    best_auc = (ckpt_cb.best_model_score.item()
                if ckpt_cb.best_model_score is not None else 0.0)
    return best_model, best_auc


@torch.no_grad()
def evaluate_on_test(models, test_sessions, device):
    transform = models[0].transform
    patch_grid = models[0].patch_grid

    test_ds = SessionDataset(
        test_sessions, transform, patch_grid, slices_per_step=None,
    )

    all_probs, all_labels = [], []

    for i in range(len(test_ds)):
        images, gaze_weights, label = test_ds[i]
        images = images.to(device)
        gaze_weights = gaze_weights.to(device)

        fold_probs = []
        for model in models:
            model.eval().to(device)
            logits = model(images, gaze_weights)  # (1, C)
            probs = torch.softmax(logits, dim=-1)
            fold_probs.append(probs[0, 1].cpu().item())
            model.to('cpu')

        all_probs.append(np.mean(fold_probs))
        all_labels.append(label.item())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    threshold = youdens_j_threshold(all_labels, all_probs)
    metrics = compute_metrics(all_labels, all_probs, threshold)
    return all_probs, all_labels, threshold, metrics


def youdens_j_threshold(y_true, probs):
    thresholds = np.linspace(0, 1, 201)
    best_j, best_t = -1, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, preds, labels=[0, 1],
        ).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def compute_metrics(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, preds, labels=[0, 1],
    ).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(y_true, preds)
    return {'ROC_AUC': auc, 'F1': f1, 'Sensitivity': sens, 'Specificity': spec}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end DINOv2 Gaze-Weighted Classifier',
    )
    parser.add_argument(
        '--pooling', choices=['gaze_weighted', 'cls', 'mean_patch'],
        default='gaze_weighted',
        help='Output pooling strategy (default: gaze_weighted).',
    )
    parser.add_argument(
        '--attn-mode', choices=['none', 'gaze_bias', 'fixation_mask'],
        default='none',
        help='Gaze attention mode inside transformer (default: none).',
    )
    parser.add_argument(
        '--fixation-threshold', type=float, default=FIXATION_THRESHOLD,
        help=f'Threshold for fixation_mask mode (default: {FIXATION_THRESHOLD}).',
    )
    parser.add_argument(
        '--slices-per-step', type=int, default=SLICES_PER_STEP,
        help=f'Slices sampled per training step (default: {SLICES_PER_STEP}).',
    )
    parser.add_argument(
        '--freeze-backbone', action='store_true',
        help='Freeze DINOv2 backbone (default: unfrozen).',
    )
    args = parser.parse_args()

    slices_per_step = args.slices_per_step

    pl.seed_everything(SEED)

    train_pool, test_set = discover_sessions()
    print(f"  Train pool: {len(train_pool)} sessions  "
          f"(Expert={sum(s['label'] == 1 for s in train_pool)}, "
          f"Novice={sum(s['label'] == 0 for s in train_pool)})")
    print(f"  Test set:   {len(test_set)} sessions  "
          f"(Expert={sum(s['label'] == 1 for s in test_set)}, "
          f"Novice={sum(s['label'] == 0 for s in test_set)})")

    model_kwargs = dict(
        pooling=args.pooling,
        attn_mode=args.attn_mode,
        fixation_threshold=args.fixation_threshold,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"Attn mode: {args.attn_mode}")
    print(f"Pooling: {args.pooling}")
    if args.attn_mode == 'fixation_mask':
        print(f"Fix. threshold: {args.fixation_threshold}")
    print(f"Backbone frozen: {args.freeze_backbone}")
    print(f"Slices/step: {slices_per_step}")
    print(f"Backbone LR: {BACKBONE_LR}")
    print(f"Head LR: {LR}")

    print(f"\n {N_FOLDS}-Fold Stratified CV")
    labels = [s['label'] for s in train_pool]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_models = []
    fold_val_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(range(len(train_pool)), labels)):
        train_sessions = [train_pool[i] for i in tr_idx]
        val_sessions = [train_pool[i] for i in va_idx]

        print(f"\n Fold {fold + 1}/{N_FOLDS}: "
              f"train={len(train_sessions)}, val={len(val_sessions)}")
        t0 = time.time()

        model, val_auc = train_one_fold(
            train_sessions, val_sessions, fold, model_kwargs,
            slices_per_step=slices_per_step,
        )
        fold_models.append(model)
        fold_val_aucs.append(val_auc)

        print(f" Fold {fold + 1}/{N_FOLDS}  val AUC = {val_auc:.4f}  "
              f"({time.time() - t0:.0f}s)")

    print(f"\n Mean val AUC = {np.mean(fold_val_aucs):.4f} "
          f"± {np.std(fold_val_aucs):.4f}")

    print("\n Test Set Evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, threshold, metrics = evaluate_on_test(
        fold_models, test_set, device,
    )
    print(f"Youden's J threshold = {threshold:.3f}")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")


if __name__ == '__main__':
    main()
