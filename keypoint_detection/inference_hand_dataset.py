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
        categories_list[category["id"]] = category["name"]
    return categories_list


if __name__ == "__main__":
    image = cv2.imread("../data/multiview_hand_pose_dataset/annotated_frames/data_1/0_webcam_1.jpg")

    cfg = get_cfg()
    cfg.merge_from_file("./configs/train_rcnn_fpn.yaml")
    cfg.MODEL.WEIGHTS = "./logs/test/3/model_final.pth"

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # metadata.thing_classes = get_categories("./labels/hands.json")

    predictor = DefaultPredictor(cfg)
    predictions = predictor(image)
    predictions["instances"] = predictions["instances"].to("cpu")

    get_top_x_predictions(predictions, 1)
    print(predictions["instances"].pred_keypoints)
    for i in range(21):
        image = cv2.circle(image, (int(predictions["instances"].pred_keypoints[0, i, 0]), int(predictions["instances"].pred_keypoints[0, i, 1])), radius=3, color=(0, 0, 255), thickness=-1)
    # get_detections_above_confidence(predictions, 0.9)

    # labels = class_id2label(predictions["instances"].pred_classes, metadata.thing_classes)
    # scores = predictions["instances"].scores.numpy()
    # view_strings = combine_label_and_score(labels, scores)

    visualizer = Visualizer(image, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    visualized_detection = visualizer.overlay_instances(
        keypoints=predictions["instances"].pred_keypoints,
        boxes=predictions["instances"].pred_boxes,
        # labels=view_strings
        ).get_image()

    plt.imshow(cv2.cvtColor(visualized_detection, cv2.COLOR_BGR2RGB))
    plt.show()