import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt

from tqdm import tqdm


def raw_to_coco(dirpath):
    filenames_ = os.listdir(dirpath)
    filenames = []
    for filename in filenames_:
        if filename[-6:] == "_l.jpg":
            filenames.append(filename[:-6])

    num_keypoints = 21

    coco_json = {}
    coco_json["images"] = []
    coco_json["annotations"] = []
    coco_json["categories"] = []

    category_field = {}
    category_field["keypoints"] = [i for i in range(num_keypoints)]
    category_field["id"] = 1
    category_field["name"] = "hand"

    coco_json["categories"].append(category_field)

    image_id = 0
    annotation_id = 1
    
    for filename in tqdm(filenames, total=len(filenames)):

        image = cv2.imread(os.path.join(dirpath, filename + '_l.jpg'))
        image_id += 1

        image_field = {}
        image_field["id"] = image_id
        image_field["width"] = image.shape[1]
        image_field["height"] = image.shape[0]
        image_field["file_name"] = filename + '_l.jpg'

        coco_json["images"].append(image_field)


        with open(os.path.join(dirpath, filename + '_l.json'), 'r') as file:
            raw_labels_l = json.load(file)

        keypoints = []
        handpoints = np.array(raw_labels_l['hand_pts'])

        for i in range(num_keypoints):
            keypoints.append(int(handpoints[i, 0]))
            keypoints.append(int(handpoints[i, 1]))
            keypoints.append(2)

        annotation_field = {}
        annotation_field["keypoints"] = keypoints
        annotation_field["num_keypoints"] = num_keypoints
        annotation_field["id"] = annotation_id
        annotation_field["image_id"] = image_id
        annotation_field["category_id"] = 1
        annotation_field["iscrowd"] = 0

        x_min, y_min = int(np.amin(handpoints[:, 0])), int(np.amin(handpoints[:, 1]))
        x_max, y_max = int(np.amax(handpoints[:, 0])), int(np.amax(handpoints[:, 1]))

        width, height = x_max - x_min, y_max - y_min
        annotation_field["bbox"] = [x_min, y_min, width, height]
        annotation_field["area"] = width * height

        coco_json["annotations"].append(annotation_field)

        annotation_id += 1


        if not os.path.exists(os.path.join(dirpath, filename + '_r.json')):
            continue 

        with open(os.path.join(dirpath, filename + '_r.json'), 'r') as file:
            raw_labels_r = json.load(file)

        keypoints = []
        handpoints = np.array(raw_labels_r['hand_pts'])

        for i in range(num_keypoints):
            keypoints.append(int(handpoints[i, 0]))
            keypoints.append(int(handpoints[i, 1]))
            keypoints.append(2)

        annotation_field = {}
        annotation_field["keypoints"] = keypoints
        annotation_field["num_keypoints"] = num_keypoints
        annotation_field["id"] = annotation_id
        annotation_field["image_id"] = image_id
        annotation_field["category_id"] = 1
        annotation_field["iscrowd"] = 0

        x_min, y_min = int(np.amin(handpoints[:, 0])), int(np.amin(handpoints[:, 1]))
        x_max, y_max = int(np.amax(handpoints[:, 0])), int(np.amax(handpoints[:, 1]))

        width, height = x_max - x_min, y_max - y_min
        annotation_field["bbox"] = [x_min, y_min, width, height]
        annotation_field["area"] = width * height

        coco_json["annotations"].append(annotation_field)

        annotation_id += 1

        # if id_ == 150:
        #     for i in range(21):
        #         cv2.circle(image, (keypoints[i*3+0], keypoints[i*3+1]), 3, (255, 0, 0), 1)
        #         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        #     plt.imshow(image)
        #     plt.show()
        #     exit()

    return coco_json

if __name__ == "__main__":
    coco_json = raw_to_coco("/home/casey/Downloads/hand_labels/manual_train")

    with open("../../data/hands_labels.json", 'w') as file:
        json.dump(coco_json, file)