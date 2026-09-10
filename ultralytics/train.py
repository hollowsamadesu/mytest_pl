#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load a model
#model = YOLO("/data1/zhaohaowei/peak_detection/pltest/mytest_pl/ultralytics/yolo11n.pt")  # load a pretrained model (recommended for training)

#model = YOLO("/data1/zhaohaowei/peak_detection/yolo/ultralytics/runs/segment/seg-with-pre-train/weights/best.pt")  # load a pretrained model (recommended for training)
#model = YOLO("/data1/zhaohaowei/peak_detection/yolo/ultralytics/yolo11m_pre_new.pt") #加载用其他峰数据预训练的数据集
model = YOLO("yolo11m.yaml") #从头训练模型

# Train the model with 2 GPUs
results = model.train(data="dataset.yaml",batch=128, epochs=200, patience=150, imgsz=640, device=[0,1,2,3,4,5,6,7], optimizer = "SGD")