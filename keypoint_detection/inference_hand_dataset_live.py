import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import json

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def get_top_x_predictions(predictions, x):
    predictions["instances"] = predictions["instances"][:x]

def get_detections_above_confidence(predictions, conf=0.9):
    scores = predictions["instances"].scores.cpu().numpy()
    x = len(scores) - np.searchsorted(scores[::-1], conf)
    get_top_x_predictions(predictions, x)

def class_id2label(ids, classes):
    labels = []
    for id in ids:
        labels.append(classes[id])
    return labels

def combine_label_and_score(labels, scores):
    view_strings = []
    for label, score in zip(labels, scores):
        view_str = f"{label} - {score:.3f}"
        view_strings.append(view_str)
    return view_strings

def get_categories(json_path):
    with open(json_path, 'r') as file:
        coco = json.load(file)
    categories_list = ["" for i in range(len(coco["categories"]))]
    for category in coco["categories"]:
        categories_list[category["id"]-1] = category["name"]
    return categories_list


if __name__ == "__main__":

    cfg = get_cfg()
    cfg.merge_from_file("./configs/train_rcnn_fpn.yaml")
    cfg.MODEL.WEIGHTS = "./logs/multiview_test/example/model_final.pth"

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    metadata.thing_classes = get_categories("../data/hands_labels.json")
    metadata.keypoint_names = [
        "F4_KNU1_A",
        "F4_KNU1_B",
        "F4_KNU2_A",
        "F4_KNU3_A",
        "F3_KNU1_A",
        "F3_KNU1_B",
        "F3_KNU2_A",
        "F3_KNU3_A",
        "F1_KNU1_A",
        "F1_KNU1_B",
        "F1_KNU2_A",
        "F1_KNU3_A",
        "F2_KNU1_A",
        "F2_KNU1_B",
        "F2_KNU2_A",
        "F2_KNU3_A",
        "TH_KNU1_A",
        "TH_KNU1_B",
        "TH_KNU2_A",
        "TH_KNU3_A",
        "PALM_POSITION",
    ]
    metadata.keypoint_connection_rules = [
        ("F4_KNU1_B", "F4_KNU1_A", (255, 0, 0)),
        ("F4_KNU1_A", "F4_KNU2_A", (0, 255, 0)),
        ("F4_KNU2_A", "F4_KNU3_A", (0, 0, 255)),

        ("F3_KNU1_B", "F3_KNU1_A", (255, 0, 0)),
        ("F3_KNU1_A", "F3_KNU2_A", (0, 255, 0)),
        ("F3_KNU2_A", "F3_KNU3_A", (0, 0, 255)),

        ("F1_KNU1_B", "F1_KNU1_A", (255, 0, 0)),
        ("F1_KNU1_A", "F1_KNU2_A", (0, 255, 0)),
        ("F1_KNU2_A", "F1_KNU3_A", (0, 0, 255)),

        ("F2_KNU1_B", "F2_KNU1_A", (255, 0, 0)),
        ("F2_KNU1_A", "F2_KNU2_A", (0, 255, 0)),
        ("F2_KNU2_A", "F2_KNU3_A", (0, 0, 255)),

        ("TH_KNU1_B", "TH_KNU1_A", (255, 0, 0)),
        ("TH_KNU1_A", "TH_KNU2_A", (0, 255, 0)),
        ("TH_KNU2_A", "TH_KNU3_A", (0, 0, 255)),
    ]
    print(metadata.keypoint_connection_rules)
    print(metadata.keypoint_names)

    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()  # Read an frame from the webcam.

        predictor = DefaultPredictor(cfg)
        predictions = predictor(image)
        predictions["instances"] = predictions["instances"].to("cpu")

        get_top_x_predictions(predictions, 1)

        labels = class_id2label(predictions["instances"].pred_classes, metadata.thing_classes)
        scores = predictions["instances"].scores.numpy()
        view_strings = combine_label_and_score(labels, scores)

        visualizer = Visualizer(image, metadata=metadata)
        visualized_detection = visualizer.overlay_instances(
            keypoints=predictions["instances"].pred_keypoints,
            boxes=predictions["instances"].pred_boxes,
            labels=view_strings
            ).get_image()

        cv2.imshow('frame', visualized_detection)  # While we're here, we might as well show it on the screen.

        # Close the script when q is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
