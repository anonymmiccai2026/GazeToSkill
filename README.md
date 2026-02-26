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
