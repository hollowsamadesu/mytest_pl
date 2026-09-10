# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.tasks import SemiDetectionModel
from ultralytics.utils import DEFAULT_CFG, ops


class SemiDetectionPredictor(DetectionPredictor):
    """
    Predictor for semi-supervised detection model (no mask branch).
    Extends DetectionPredictor; postprocess is identical to standard detection.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the SemiDetectionPredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "detect"

    def postprocess(self, preds, img, orig_imgs):
        """Applies NMS and returns Results with boxes only (no masks)."""
        p = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
            else:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
        return results
