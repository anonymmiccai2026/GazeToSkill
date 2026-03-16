import os
import json
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple

BASE_PATH = ''
LUNG_CTS_DIR = os.path.join(BASE_PATH, 'lung_CTs')

RADIOLOGIST_FOLDERS = {
    'A': 'folderA',
    'B': 'folderB',
    'C': 'folderC',
    'D': 'folderD',
    'E': 'folderE',
}

CT_SLICE_SIZE = 512

SCREEN_WIDTH_CM = 60
SCREEN_WIDTH_PX = 2560
FOVEAL_DEG = 1  
DEFAULT_SIGMA_CT_PX = 17.0

OUTPUT_SUBDIR = 'folder'


def degrees_to_pixels(degrees, screen_width_cm, screen_width_px, viewing_distance_cm):
    cm = viewing_distance_cm * np.tan(np.radians(degrees))
    pixels = cm * (screen_width_px / screen_width_cm)
    return pixels


def compute_sigma_from_final_csv(final_csv_path):
    df = pd.read_csv(final_csv_path,
                     usecols=['x', 'x_ct', 'left_z', 'right_z'])

    # Viewing distance
    eye_dist_mm = (df['left_z'] + df['right_z']) / 2
    viewing_distance_cm = eye_dist_mm.mean() / 10

    valid = df[(df['x_ct'] >= 0) & (df['x_ct'] <= CT_SLICE_SIZE)].copy()

    if len(valid) < 10:
        return DEFAULT_SIGMA_CT_PX

    dx_screen = valid['x'].diff().abs()
    dx_ct = valid['x_ct'].diff().abs()
    mask = (dx_screen > 5) & (dx_ct > 1)

    if mask.sum() < 5:
        return DEFAULT_SIGMA_CT_PX

    screen_to_ct_ratio = (dx_ct[mask] / dx_screen[mask]).median()

    screen_px = degrees_to_pixels(FOVEAL_DEG, SCREEN_WIDTH_CM,
                                  SCREEN_WIDTH_PX, viewing_distance_cm)
    sigma_ct_px = screen_px * screen_to_ct_ratio

    if sigma_ct_px < 5 or sigma_ct_px > 50:
        return DEFAULT_SIGMA_CT_PX

    return sigma_ct_px

def find_ct_folder(lung_cts_dir, ct_id):
    for patient_num in os.listdir(lung_cts_dir):
        patient_path = os.path.join(lung_cts_dir, patient_num, "LIDC-IDRI")
        if os.path.isdir(patient_path):
            if ct_id in os.listdir(patient_path):
                return os.path.join(patient_path, ct_id)
    return None


def find_nifti_file(ct_folder: str) -> Optional[str]:
    nii_files = glob.glob(os.path.join(ct_folder, '*.nii.gz'))
    if len(nii_files) == 1:
        return nii_files[0]
    elif len(nii_files) > 1:
        for f in nii_files:
            basename = os.path.basename(f).lower()
            if 'nodule' not in basename and 'seg' not in basename:
                return f
        return nii_files[0]
    return None


def load_ct_volume(nifti_path):
    ct_data = nib.load(nifti_path).get_fdata()
    ct_data_rotated = np.flip(np.rot90(ct_data, k=-1, axes=(0, 1)), axis=1)
    return ct_data_rotated

def build_single_heatmap(slice_fixations,
                         image_shape,
                         sigma):
    fix_map = np.zeros(image_shape, dtype=np.uint8)
    for _, row in slice_fixations.iterrows():
        x = int(round(row['x_ct']))
        y = int(round(row['y_ct']))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            fix_map[y, x] = 255

    heatmap = gaussian_filter(fix_map.astype(np.float64), sigma=sigma)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def create_heatmaps_config1(fixation_df,
                            n_slices,
                            sigma,
                            image_shape
                            ):
    heatmaps = []
    indices = []

    for slice_idx in range(n_slices):
        slice_fixations = fixation_df[fixation_df['SLICE_NUMBER'] == slice_idx]
        if slice_fixations.empty:
            continue

        heatmaps.append(build_single_heatmap(slice_fixations, image_shape, sigma))
        indices.append(slice_idx)

    if not heatmaps:
        return np.zeros((0, *image_shape), dtype=np.float64), np.array([], dtype=np.int64)

    return np.stack(heatmaps, axis=0), np.array(indices, dtype=np.int64)


def create_heatmaps_config2(fixation_df,
                            n_slices,
                            sigma,
                            image_shape
                            ):
    heatmap_stack = np.zeros((n_slices, image_shape[0], image_shape[1]),
                             dtype=np.float64)

    for slice_idx in range(n_slices):
        slice_fixations = fixation_df[fixation_df['SLICE_NUMBER'] == slice_idx]
        if slice_fixations.empty:
            heatmap_stack[slice_idx] = np.zeros(image_shape, dtype=np.uint8)
            continue

        heatmap_stack[slice_idx] = build_single_heatmap(
            slice_fixations, image_shape, sigma)

    return heatmap_stack

def process_record(record_dir,
                   lung_cts_dir):
    metadata_path = os.path.join(record_dir, 'metadata.json')
    csv_path = os.path.join(record_dir, 'fixation.csv')
    final_csv_path = os.path.join(record_dir, 'final.csv')

    if not os.path.exists(csv_path):
        print(f"SKIP (no fixation.csv): {record_dir}")
        return False
    if not os.path.exists(metadata_path):
        print(f"SKIP (no metadata.json): {record_dir}")
        return False
    if not os.path.exists(final_csv_path):
        print(f"SKIP (no final.csv): {record_dir}")
        return False

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    ct_id = metadata['CT_ID']
    rad = metadata.get('rad', '')
    folder_name = metadata.get('folder_name', os.path.basename(record_dir))

    # Find CT
    ct_folder = find_ct_folder(lung_cts_dir, ct_id)
    if ct_folder is None:
        print(f"SKIP (CT not found for {ct_id}): {record_dir}")
        return False
    nifti_path = find_nifti_file(ct_folder)
    if nifti_path is None:
        print(f"SKIP (no .nii.gz for {ct_id}): {record_dir}")
        return False

    sigma = compute_sigma_from_final_csv(final_csv_path)

    # Load fixation data
    fix_df = pd.read_csv(csv_path)
    fix_df = fix_df[fix_df['CT_plane'] == 'A'].copy()
    if fix_df.empty:
        print(f"  SKIP (no axial fixations): {record_dir}")
        return False

    # Load CT to get number of slices
    volume = load_ct_volume(nifti_path)
    n_slices = volume.shape[2]

    # Config 1: only fixated slices
    heatmaps_c1, slice_indices_c1 = create_heatmaps_config1(fix_df, n_slices, sigma)

    # Config 2: all slices, zeros for empty
    heatmaps_c2 = create_heatmaps_config2(fix_df, n_slices, sigma)

    output_dir = os.path.join(record_dir, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'heatmaps_1vis_degree.npy'), heatmaps_c1)
    np.save(os.path.join(output_dir, 'slice_indices_1vis_degree.npy'), slice_indices_c1)
    np.save(os.path.join(output_dir, 'heatmaps_1vis_degree_full.npy'), heatmaps_c2)

    out_metadata = {
        'rad': rad,
        'ct_id': ct_id,
        'folder_name': folder_name,
        'n_slices': n_slices,
        'n_fixated_slices': len(slice_indices_c1),
        'sigma_ct_px': round(float(sigma), 2),
        'foveal_deg': FOVEAL_DEG,
        'nifti_path': nifti_path,
        'record_dir': record_dir,
    }
    with open(os.path.join(output_dir, 'metadata_1deg.json'), 'w') as f:
        json.dump(out_metadata, f, indent=2)

    return True


def process_all(base_path,
                lung_cts_dir,
                radiologist_folders):
    total_ok = 0
    total_skip = 0

    for rad_id, folder_name in radiologist_folders.items():
        rad_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(rad_path):
            print(f"Folder not found for Rad {rad_id}: {rad_path}")
            continue

        record_dirs = sorted([
            os.path.join(rad_path, d)
            for d in os.listdir(rad_path)
            if os.path.isdir(os.path.join(rad_path, d))
        ])

        print(f"\nRadiologist {rad_id} ({folder_name}): {len(record_dirs)} records")

        for record_dir in record_dirs:
            success = process_record(record_dir, lung_cts_dir)
            if success:
                print(f"  OK: {os.path.basename(record_dir)}")
                total_ok += 1
            else:
                total_skip += 1

    print(f"\nDone. Processed: {total_ok}, Skipped: {total_skip}")


def visualize_sample(save_path='output.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sample_record = os.path.join(BASE_PATH, RADIOLOGIST_FOLDERS['A'],
                                 '2025-08-05_13-30-09')

    with open(os.path.join(sample_record, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    ct_id = metadata['CT_ID']
    ct_folder = find_ct_folder(LUNG_CTS_DIR, ct_id)
    nifti_path = find_nifti_file(ct_folder)
    volume = load_ct_volume(nifti_path)
    n_slices = volume.shape[2]

    fix_df = pd.read_csv(os.path.join(sample_record, 'fixation.csv'))
    fix_df = fix_df[fix_df['CT_plane'] == 'A'].copy()

    sigma = compute_sigma_from_final_csv(
        os.path.join(sample_record, 'final.csv'))

    print(f"Record: {sample_record}")
    print(f"CT_ID: {ct_id}")
    print(f"CT shape: {volume.shape}")
    print(f"Sigma: {sigma:.1f} CT pixels")
    print(f"Fixations: {len(fix_df)} rows (axial)")

    # Config 1
    heatmaps_c1, indices_c1 = create_heatmaps_config1(fix_df, n_slices, sigma)
    print(f"Config 1: {heatmaps_c1.shape} (only fixated slices)")

    # Config 2
    heatmaps_c2 = create_heatmaps_config2(fix_df, n_slices, sigma)
    print(f"Config 2: {heatmaps_c2.shape} (all slices, zeros for empty)")

    # Pick 5 fixated slices for visualization
    slices_with_fixations = sorted(fix_df['SLICE_NUMBER'].unique())
    n_show = min(5, len(slices_with_fixations))
    idx = np.linspace(0, len(slices_with_fixations) - 1, n_show, dtype=int)
    show_slices = [int(slices_with_fixations[i]) for i in idx]

    def window_ct(s, wl=-600, ww=1500):
        lo, hi = wl - ww / 2, wl + ww / 2
        return np.clip((s - lo) / (hi - lo), 0, 1)

    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))

    for col, sl in enumerate(show_slices):
        ct_slice = window_ct(volume[:, :, sl])
        slice_fix = fix_df[fix_df['SLICE_NUMBER'] == sl]

        axes[0, col].imshow(ct_slice, cmap='gray', vmin=0, vmax=1)
        if not slice_fix.empty:
            axes[0, col].scatter(slice_fix['x_ct'], slice_fix['y_ct'],
                                 c='red', s=15, alpha=0.7, edgecolors='none')
        axes[0, col].set_title(f'Slice {sl}', fontsize=12)
        axes[0, col].axis('off')

        c1_idx = np.where(indices_c1 == sl)[0]
        if len(c1_idx) > 0:
            hm1 = heatmaps_c1[c1_idx[0]]
        else:
            hm1 = np.zeros((CT_SLICE_SIZE, CT_SLICE_SIZE))
        axes[1, col].imshow(ct_slice, cmap='gray', vmin=0, vmax=1)
        axes[1, col].imshow(hm1, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, col].set_title(f'C1: {sl}', fontsize=12)
        axes[1, col].axis('off')

        axes[2, col].imshow(ct_slice, cmap='gray', vmin=0, vmax=1)
        axes[2, col].imshow(heatmaps_c2[sl], cmap='jet', alpha=0.5,
                            vmin=0, vmax=1)
        axes[2, col].set_title(f'C2: {sl}', fontsize=12)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel('CT + Fixations', fontsize=13)
    axes[1, 0].set_ylabel('Config 1 (fixated only)', fontsize=13)
    axes[2, 0].set_ylabel('Config 2 (all slices)', fontsize=13)

    fig.suptitle(
        f'{ct_id}  |  Rad A  |  sigma={sigma:.1f}px (1 deg)  |  '
        f'{n_slices} total  |  {len(slices_with_fixations)} fixated',
        fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        visualize_sample(save_path='output.png')
    else:
        print("=" * 60)
        print("Heatmap Stack Generation — 1 deg foveal vision")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"Base path: {BASE_PATH}")
        print(f"Lung CTs dir: {LUNG_CTS_DIR}")
        print(f"Slice size: {CT_SLICE_SIZE}x{CT_SLICE_SIZE}")
        print(f"Foveal angle: {FOVEAL_DEG} deg (sigma computed per session)")
        print(f"Output subdir: {OUTPUT_SUBDIR}")
        print(f"Radiologists: {list(RADIOLOGIST_FOLDERS.keys())}")

        process_all()
