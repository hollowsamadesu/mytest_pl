import albumentations as A
import numpy as np
import torch
from collections import defaultdict
import cv2


class StrongNoiseBlurAug:
    """Strong augmentation for detection: GaussianBlur + BBoxSafeRandomCrop + CoarseDropout + Resize.
    No mask synchronization needed (detection-only).
    """

    def __init__(self):
        self.transform = A.ReplayCompose([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.BBoxSafeRandomCrop(erosion_rate=0.0),
            A.CoarseDropout(min_holes=3,
                            max_holes=6,
                            min_height=10,
                            max_height=20,
                            min_width=10,
                            max_width=20,
                            fill_value=0,
                            p=0.8),
            A.Resize(height=640, width=640),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.6,
            label_fields=['cls', 'pos_indice', 'label_idx'],
            filter_invalid_bboxes=True
        ))

    def __call__(self, batch):
        images = batch["img"]
        bboxes = batch["bboxes"]
        box_cls = batch["cls"]
        batch_idx = batch["batch_idx"].tolist()
        device = images.device
        images_augs = []

        for i in range(batch["img"].shape[0]):
            image_aug = defaultdict(lambda: {
                "img": [],
                "bboxes": [],
                "cls": [],
                "idx": []
            })
            image_aug["img"] = images[i]
            indice = [j for j, x in enumerate(batch_idx) if x == i]
            image_aug["boxes"] = bboxes[indice]
            image_aug["cls"] = box_cls[indice]
            image_aug['idx'] = []
            images_augs.append(image_aug)

        idx_count = 0
        for item in images_augs:
            img_np = item["img"].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np
            bboxes_np = item["boxes"].cpu().numpy()
            boxes_np = np.array([box for box in bboxes_np])
            cls_np = item['cls'].cpu().numpy()
            clses_np = np.array([cls for cls in cls_np])

            if boxes_np.shape[0] > 0:
                label_idx = batch_idx[idx_count:idx_count + boxes_np.shape[0]]
                idx_count += boxes_np.shape[0]
                label_idx = np.array(label_idx)
                pos_indices = np.array(range(boxes_np.shape[0]))
                transformed = self.transform(
                    image=img_np, bboxes=boxes_np,
                    cls=clses_np, pos_indice=pos_indices, label_idx=label_idx
                )
                item["boxes"] = torch.tensor(transformed["bboxes"], device=device)
                item['cls'] = torch.tensor(transformed['cls'], device=device).view(-1, 1)
                item['idx'] = torch.tensor(transformed['label_idx'], device=device)
            else:
                transformed = self.transform(
                    image=img_np, bboxes=np.empty((0, 4)),
                    cls=np.empty(0), pos_indice=np.empty(0), label_idx=np.empty(0)
                )

            aug_img = transformed["image"]
            aug_img = torch.from_numpy(aug_img).float().permute(2, 0, 1) / 255.0
            item["img"] = aug_img

        new_images = []
        new_bboxes = torch.empty(0, 4).to(device)
        new_cls = torch.empty(0, 1).to(device)
        new_idx = torch.empty(0).to(device)

        for item in images_augs:
            new_images.append(item["img"])
            new_bboxes = torch.cat([new_bboxes, item["boxes"]])
            new_cls = torch.cat([new_cls, item['cls']])
            if not isinstance(item['idx'], list):
                new_idx = torch.cat([new_idx, item['idx']])

        batch["img"] = torch.stack(new_images)
        batch["bboxes"] = new_bboxes
        batch['cls'] = new_cls
        batch['batch_idx'] = new_idx

        return batch


class WeekAugmentation:
    """Weak augmentation: HorizontalFlip + GaussianBlur + Resize.
    Records whether horizontal flip was applied for pseudo-label correction.
    """

    def __init__(self):
        self.transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.2),
            A.Resize(height=640, width=640),
        ], save_key="replay")

    def __call__(self, batch):
        images = batch["img"]
        images_aug = []
        is_horizons = []

        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np
            aug_dict = self.transform(image=img_np)
            aug = aug_dict["image"]
            replay = aug_dict["replay"]
            is_horizon = check_horizon(replay)
            aug = torch.from_numpy(aug).float().permute(2, 0, 1) / 255.0
            images_aug.append(aug)
            is_horizons.append(is_horizon)

        batch["img"] = torch.stack(images_aug)
        batch["aug_type"] = is_horizons
        return batch, is_horizons


def check_horizon(replay):
    """Check if horizontal flip was applied in the augmentation replay."""
    transforms = replay.get('transforms', {})
    for t in transforms:
        if t.get('__class_fullname__') == "HorizontalFlip":
            return t.get('applied', False)
    return False


class HorizonFlip:
    """Callback flip for pseudo-label bboxes when weak augmentation applied horizontal flip."""

    def __init__(self):
        self.transform = A.Compose(
            [A.HorizontalFlip(p=1.0), A.Resize(height=640, width=640)],
            A.BboxParams(
                format="pascal_voc",
                min_height=0,
                min_width=0,
                min_visibility=0,
                filter_invalid_bboxes=True,
                label_fields=['pos_indices']
            ),
        )

    def __call__(self, pseudo_labels, horizons):
        """Flip bboxes in pseudo_labels when corresponding image was horizontally flipped.

        Args:
            pseudo_labels: list of Tensor[N, 6] (xyxy + conf + cls), one per image
            horizons: list of bool, whether each image was flipped
        """
        device = pseudo_labels[0].device if isinstance(pseudo_labels[0], torch.Tensor) else pseudo_labels[0][0].device
        for i in range(len(pseudo_labels)):
            label = pseudo_labels[i]
            if isinstance(label, list):
                label = label[0] if len(label) > 0 else torch.empty(0, 6, device=device)
            if horizons[i] and label.shape[0] > 0:
                boxes = label[:, :4].cpu().numpy()
                pos_indices = np.array(range(boxes.shape[0]))
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                transformed = self.transform(
                    image=dummy_image, bboxes=boxes, pos_indices=pos_indices
                )
                valid_indices = transformed["pos_indices"]
                pseudo_labels[i] = label[valid_indices].clone()
                pseudo_labels[i][:, :4] = torch.tensor(transformed["bboxes"], device=device)
        return pseudo_labels
