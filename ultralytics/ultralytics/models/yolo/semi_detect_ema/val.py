# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG, ops


class SemiDetectionValidator(DetectionValidator):
    """
    Validator for semi-supervised detection model.
    Essentially identical to DetectionValidator with task set to 'detect'.
    Supports optional teacher_model for dual-model validation.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, teacher_model=None):
        """Initialize SemiDetectionValidator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "detect"
        self.teacher_model = teacher_model
