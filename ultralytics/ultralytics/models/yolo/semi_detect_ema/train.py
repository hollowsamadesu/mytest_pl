# Ultralytics YOLO 🚀, AGPL-3.0 license
# EMA-based semi-supervised detection trainer (teacher = EMA of student)

from pathlib import Path
import os
import math
import time
import warnings
from datetime import datetime, timedelta
from copy import deepcopy
from random import random

import numpy as np
import torch
from torch import nn, optim
from torch import distributed as dist
import torch.nn.functional as F
import torchvision
import subprocess

from ultralytics.models import yolo
from ultralytics.models.yolo.semi_detect.val import SemiDetectionValidator
from ultralytics.nn.tasks import SemiDetectionModel
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import ops
from ultralytics.utils import DEFAULT_CFG, RANK, LOCAL_RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.data import build_dataloader
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.nn.autobackend import check_class_names

from ultralytics.utils import (
    LOGGER,
    TQDM,
    callbacks,
    colorstr,
    emojis,
    yaml_save,
    yaml_load,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)
from ultralytics.models.yolo.semi_detect.aug_unsup import StrongNoiseBlurAug, WeekAugmentation, HorizonFlip


class EMASemiDetectionTrainer(yolo.detect.DetectionTrainer):
    """
    EMA-based semi-supervised detection trainer.
    Student model: trained with gradients on labeled + pseudo-labeled data.
    Teacher model: EMA of student, used only for pseudo-label generation (no gradients).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "semi_detect_ema"
        super().__init__(cfg, overrides, _callbacks)

        self.ema_decay = 0.99
        self.unsup_weight = getattr(self.args, "unsup_weight", 1.0)
        self.auto_train = getattr(self.args, "self_train", False)
        self.total_loss = None
        self.lamda = 0.7
        self.strong_aug = StrongNoiseBlurAug()
        self.weak_aug = WeekAugmentation()
        self.horizonFlip = HorizonFlip()
        self.reset_teacher = False
        self.teacher_model = None  # will be created as EMA of student in _setup_train
        self.teacher_ema = None  # not used in EMA version, but needed for validator compatibility

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create and return a SemiDetectionModel."""
        model = SemiDetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a validator for student model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SemiDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks
        )

    def get_teacher_validator(self):
        """Return a validator for teacher model."""
        return SemiDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks,
            teacher_model=self.teacher_model
        )

    def plot_training_samples(self, batch, ni):
        """Plot training samples (detection-only, no masks)."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, segment=False, on_plot=self.on_plot)

    @torch.no_grad()
    def update_teacher_ema(self):
        """Update teacher model via EMA from student model (pure EMA, no optimizer)."""
        student_model = self.model.module if hasattr(self.model, "module") else self.model
        teacher_model = self.teacher_model

        min_decay = 0.99
        max_decay = 0.999
        cos_value = math.cos(math.pi * self.epoch / self.epochs)
        self.ema_decay = max_decay - 0.5 * (max_decay - min_decay) * (1 + cos_value)

        for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
            if t_param.data.shape != s_param.data.shape:
                continue
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

        # Sync buffers (e.g. BN running_mean, running_var)
        for t_buffer, s_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            if t_buffer.data.shape != s_buffer.data.shape:
                continue
            t_buffer.data.copy_(s_buffer.data)

        teacher_model.eval()

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:
            world_size = 0
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning("WARNING: 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING: 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _do_train(self, world_size=1):

        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        if self.auto_train:
            nb = len(self.unsup_loader)
        else:
            nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )

        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        epoch = self.start_epoch
        self.optimizer.zero_grad()

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self.model.train()
            self.teacher_model.eval()  # teacher always in eval mode for pseudo-label generation

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
                self.unsup_loader.sampler.set_epoch(epoch)

            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                if not self.auto_train:
                    self.train_loader.reset()
                self.unsup_loader.reset()

            if self.auto_train:
                base_iter = enumerate(self.unsup_loader)

                def wrap_unsup(iterable):
                    for i, u in iterable:
                        yield i, (None, u)

                it = wrap_unsup(base_iter)
            else:
                it = enumerate(zip(self.train_loader, self.unsup_loader))

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(it, total=nb)
            else:
                pbar = it

            self.tloss = None

            if epoch < 200:
                lamda = 0
            else:
                lamda = min((0.5 * epoch / self.epochs), 0.4)

            if self.auto_train:
                lamda = 1.0

            for i, (batch, unsup_batch) in pbar:
                self.run_callbacks("on_train_batch_start")

                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    if batch is not None:
                        batch = self.preprocess_batch(batch)
                    unsup_batch = self.preprocess_semi_batch(unsup_batch)

                    if not self.auto_train:
                        # Student model learns from labeled data
                        self.loss, self.loss_items = self.model(batch)
                        self.total_loss = self.loss
                    else:
                        self.loss = 0

                    if lamda != 0:
                        # Step 1: Teacher generates pseudo-labels (autocast DISABLED for teacher)
                        unsup_w = deepcopy(unsup_batch)
                        unsup_w, horizons = self.weak_aug(unsup_w)
                        unsup_w["img"] = unsup_w["img"].to(self.device)

                        with torch.amp.autocast("cuda", enabled=False):
                            with torch.no_grad():
                                self.teacher_model.eval()
                                pseudo_pred, _ = self.teacher_model(unsup_w)
                                pseudo_box_label = self.get_pseudo_box_label(pseudo_pred[0])
                                pseudo_label = self.get_pseudo_label(pseudo_box_label)
                                pseudo_label = self.horizonFlip(pseudo_label, horizons)

                        # Step 2: Prepare strong-augmented data (FP32, outside teacher's autocast-off)
                        unsup_s = deepcopy(unsup_batch)
                        unsup_s = align_batch(unsup_s, pseudo_label)
                        unsup_s = self.strong_aug(unsup_s)

                        # Step 3: Student learns from pseudo-labeled data (inside outer autocast)
                        unlabel_pred, _ = self.model(unsup_s)
                        self.unsup_loss, unsup_loss_items = self.model.module.new_unsup_loss(unlabel_pred, unsup_s)

                        if self.auto_train:
                            self.total_loss = self.unsup_loss
                            self.loss_items = unsup_loss_items
                            self.loss = self.unsup_loss
                        else:
                            self.total_loss = ((1 - lamda) * self.loss + (lamda * 1) * self.unsup_loss)

                    self.model.train()

                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                    # Backward pass (student only)
                self.scaler.scale(self.total_loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.update_teacher_ema()  # EMA update after each optimizer step
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            unsup_batch["img"].shape[0] if self.auto_train else batch["cls"].shape[0],
                            unsup_batch["img"].shape[-1] if self.auto_train else batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        if self.auto_train:
                            self.plot_training_samples(unsup_batch, ni)
                        else:
                            self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs

                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop or not self.auto_train:
                    self.metrics, self.fitness = self.validate()
                    self.metrics1, self.fitness1 = self.teacher_validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch = epoch + 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()

        self.model = self.model.to(self.device)
        self.best_student = None
        self.set_model_attributes()

        # Freeze layers (student only)
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.info(
                    f"WARNING: setting 'requires_grad=True' for frozen layer '{k}'."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and world_size > 1:
            dist.broadcast(self.amp, src=0)
        self.amp = bool(self.amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True, broadcast_buffers=False)
            # Teacher model is NOT wrapped in DDP (no gradients needed)

        self.first_unsup = True

        # Create teacher model as deep copy of student (EMA initialization)
        student_for_teacher = self.model.module if hasattr(self.model, "module") else self.model
        self.teacher_model = deepcopy(student_for_teacher)
        self.teacher_model = self.teacher_model.to(self.device)
        # Freeze all teacher parameters (no gradients)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()
        LOGGER.info("Teacher model initialized as deep copy of student (EMA)")

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        # Batch size
        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        self.unsup_loader = self.get_dataloader(self.unsupset, batch_size=batch_size, rank=LOCAL_RANK, mode="unsup_train")
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            self.teacher_validator = self.get_teacher_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer (student only, no teacher optimizer)
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        # Scheduler (student only)
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return ckpt

    def set_model_attributes(self):
        """Set model attributes for semi-detection."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_dataset(self):
        """Load semi-supervised dataset configs."""
        self.data, self.unsup_data = check_semi_seg_dataset(self.args.data, self.args.unsup_data)
        return self.data["train"], self.unsup_data["train"], self.data["val"] or self.data["test"]

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        imgs = batch["img"]
        batch['is_label'] = True
        if self.args.multi_scale:
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def preprocess_semi_batch(self, batch):
        """Preprocesses an unsupervised batch of images."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        batch["is_label"] = False
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def get_pseudo_box_label(self, preds):
        """Apply NMS to raw predictions and return per-image detection results."""
        conf_thres = 0.25
        iou_thres = 0.45
        max_wh = 4096

        bs = preds.shape[0]
        nc = preds.shape[1] - 4
        xc = preds[:, 4:4 + nc].amax(1) > conf_thres

        preds = preds.transpose(-1, -2)
        tmp = preds
        preds[..., :4] = xywh2xyxy(tmp[..., :4])

        output = [torch.zeros((0, 6), device=preds.device)] * bs
        max_nms = 30000

        for xi, x in enumerate(preds):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            box, cls = x.split((4, nc), 1)
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            n = x.shape[0]
            if not n:
                continue
            if n > max_nms:
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            c = x[:, 5:6] * max_wh
            scores = x[:, 4]
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            output[xi] = x[i]

        return output

    def get_pseudo_label(self, p):
        """Convert NMS results to pseudo-labels."""
        pseudo_label = []
        for i, pred in enumerate(p):
            if len(pred) == 0:
                pseudo_label.append(pred)
            else:
                pseudo_label.append(pred[:, :6])
        return pseudo_label

    def teacher_validate(self):
        """Run teacher validation on test set."""
        metrics = self.teacher_validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness


def check_semi_seg_dataset(dataset, unsup_dataset):
    """Load and validate supervised and unsupervised dataset YAML configs."""
    file = check_file(dataset)
    unsup_file = check_file(unsup_dataset)

    data = yaml_load(file, append_filename=True)
    unsup_data = yaml_load(unsup_file, append_filename=True)

    extract_dir = ""
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)
    unsup_path = Path(extract_dir or unsup_data.get("path") or Path(unsup_data.get("yaml_file", "")).parent)

    data["path"] = path
    unsup_data["path"] = unsup_path
    for k in "train", "val", "test", "minival":
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    k = "train"
    if unsup_data.get(k):
        if isinstance(unsup_data[k], str):
            y = (path / unsup_data[k]).resolve()
            if not y.exists() and unsup_data[k].startswith("../"):
                y = (path / unsup_data[k][3:]).resolve()
            unsup_data[k] = str(y)
        else:
            unsup_data[k] = [str((path / y).resolve()) for y in unsup_data[k]]

    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing. either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])

    return data, unsup_data


def align_batch(batch, labels):
    """Align pseudo-labels into batch format (detection-only: no masks)."""
    batch_idx = []
    device = batch['img'].device
    batch['cls'] = torch.empty(0, 1, device=device)
    batch['bboxes'] = torch.empty(0, 4, device=device)
    batch['batch_idx'] = torch.empty(0, device=device)
    batch['conf'] = torch.empty(0, device=device)

    for i in range(len(labels)):
        label = labels[i]
        if isinstance(label, list):
            label = label[0] if len(label) > 0 else torch.empty(0, 6, device=device)
        if label.shape[0] == 0:
            continue
        for j in range(label.shape[0]):
            batch_idx.append(i)
            batch['cls'] = torch.cat((batch['cls'], torch.zeros(1, device=device).unsqueeze(-1)))
            batch['bboxes'] = torch.cat((batch['bboxes'], label[j][:4].unsqueeze(0).to(device)))
            batch['conf'] = torch.cat((batch['conf'], label[j][4].unsqueeze(0).to(device)))

    batch['batch_idx'] = torch.tensor(batch_idx, device=device)
    return batch
