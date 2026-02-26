# GazeToSkill

Train DINOv2 Gaze-Weighted Classifier.


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



