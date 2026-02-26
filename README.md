# GazeToSkill

This repository contains the anonymous code accompanying the paper: From Gaze to Skill: Automated Assessment of Radiologist Expertise Through Eye-Tracking on Thoracic CT Scans.

## Abstract

Accurate interpretation of volumetric CT requires efficient navigation of 3D image volumes and attention to diagnostically relevant regions. While eye-tracking has been widely studied in 2D medical imaging, its use for expertise assessment in volumetric CT remains limited. We propose a gaze-informed transformer framework for automated expertise classification in thoracic CT. Using a DINOv2 ViT-B/14 backbone, radiologist fixation patterns are integrated into volumetric feature learning through (1) a learnable log-space bias in self-attention and (2) gaze-weighted pooling of patch embeddings. We evaluate our approach on 182 CT reading sessions from five radiologists with varying experience levels. On a held-out test set, the model achieves a ROC-AUC of 0.909 and F1-score of 0.861, outperforming adapted methods. These findings suggest that incorporating visual search behavior into volumetric transformers may support objective, process-based expertise assessment in radiology.

## Pretrained Checkpoints

You can download the pretrained model weights here:



## Train DINOv2 Gaze-Weighted Classifier.

Examples:
- Default settings (no gaze attention, gaze-weighted pooling, backbone unfrozen):
    ```python train.py```

- Use gaze-bias + gaze-weighted pooling (full gaze model):
    ```python train.py --attn-mode gaze_bias --pooling gaze_weighted```

- Freeze backbone, use CLS pooling, train for 50 epochs:
    ```python train.py --freeze-backbone --pooling cls --epochs 50```

- Fewer folds, fewer slices per step (faster, for debugging):
    ```python train.py --folds 3 --slices-per-step 4 --epochs 10```

- All options at a glance:
    ```python train.py --help```

### Software and Dependencies

| Package           | Version       |
|-------------------|---------------|
| Python            | 3.9.13        |
| PyTorch           | 2.7.0+cu126   |
| CUDA (build)      | 12.6          |
| PyTorch Lightning | 2.6.0         |
| timm              | 1.0.24        |
| NumPy             | 2.0.2         |
| scikit-learn      | 1.6.1         |
| nibabel           | 5.3.2         |
| Pillow            | 11.2.1        |




## Evaluation on Held-Out Test Set

We evaluate our proposed model against adapted models. Performance is reported using **ROC-AUC, F1-score, sensitivity, and specificity**.

* **SGP**: Spatial Gaze Map
* **GTI**: Gaze Trajectory Image
* **FO-CT+Gaze**: Fixation-ordered CT slices with gaze overlay

### Results

| Model                           | Input                | Prediction   | ROC-AUC    | F1         | Sens.      | Spec.      |
| ------------------------------- | -------------------- | ------------ | ---------- | ---------- | ---------- | ---------- |
| TF-CNN (Sharma et al., 2021)    | SGP, GTI             | Skill        | 0.7793     | 0.8372     | **0.9231** | 0.6071     |
| TF-CNN (Sharma et al., 2021)    | FO-CT+Gaze, SGP, GTI | Skill        | 0.7454     | 0.7568     | 0.7179     | 0.7500     |
| IF-CNN (Sharma et al., 2021)    | SGP, GTI             | Skill        | 0.7308     | 0.6866     | 0.5897     | 0.8214     |
| IF-CNN (Sharma et al., 2021)    | FO-CT+Gaze, SGP, GTI | Skill        | 0.7363     | 0.7532     | 0.7436     | 0.6786     |
| LF-CNN (Sharma et al., 2021)    | SGP, GTI             | Skill        | 0.7463     | 0.7838     | 0.7436     | 0.7857     |
| LF-CNN (Sharma et al., 2021)    | FO-CT+Gaze, SGP, GTI | Skill        | 0.7637     | 0.8205     | 0.8205     | 0.7500     |
| HF-CNN (Sharma et al., 2021)    | SGP, GTI             | Skill        | 0.7518     | 0.7838     | 0.7436     | 0.7857     |
| HF-CNN (Sharma et al., 2021)    | FO-CT+Gaze, SGP, GTI | Skill        | 0.7363     | 0.7733     | 0.7436     | 0.7500     |
| Lou et al. (2023)               | FO-CT session-level  | Saliency map | 0.6782     | 0.7123     | 0.6667     | 0.7241     |
| Lou et al. (2023)               | FO-CT chunk-level    | Saliency map | 0.5570     | 0.6000     | 0.5385     | 0.6552     |
| CT-Searcher (Pham et al., 2025) | CT volume            | Scanpath     | 0.8750     | 0.8421     | 0.8205     | 0.8214     |
| **Ours**                        | CT volume            | Skill        | **0.9089** | **0.8611** | 0.7949     | **0.9310** |



