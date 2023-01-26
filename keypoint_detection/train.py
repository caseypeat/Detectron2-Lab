import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

from detectron2.evaluation import COCOEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator("hands_dataset", output_dir="logs/multiview_test/3", kpt_oks_sigmas=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


if __name__ == "__main__":

    cfg = get_cfg()
    cfg.merge_from_file("./configs/train_rcnn_fpn.yaml")
    cfg.OUTPUT_DIR = "logs/multiview_test/3"

    register_coco_instances(
        name="hands_dataset",
        metadata={},
        json_file="../data/multiview_hands.json",
        image_root="/home/casey/Downloads/multiview_hand")

    MetadataCatalog.get("hands_dataset").keypoint_names = [
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
        # "PALM_NORMAL",
    ]
    MetadataCatalog.get("hands_dataset").keypoint_flip_map = []

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()