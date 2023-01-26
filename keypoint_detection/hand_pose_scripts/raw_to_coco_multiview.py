import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt

from tqdm import tqdm


def raw_to_coco(root_path):

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

    id_ = 0

    for i in range(1, 22):
        dir_name = f"data_{i}"
        dir_path = os.path.join(root_path, dir_name)

        image_filenames = []
        label_filenames = []
        for filename in os.listdir(dir_path):
            if filename[-4:] == ".jpg":
                image_filename = filename

                scene, _, image = image_filename[:-4].split("_")
                label_filename = f"{scene}_jointsCam_{image}.txt"

                if os.path.exists(os.path.join(dir_path, label_filename)):
                    image_filenames.append(image_filename)
                    label_filenames.append(label_filename)



        # for image_filename in image_filenames:
            # scene, _, image = image_filename[:-4].split("_")
            # label_filename = f"{scene}_jointsCam_{image}.txt"
            # label_filenames.append(label_filename)

        for image_filename, label_filename in zip(image_filenames, label_filenames):
            with open(os.path.join(dir_path, label_filename), 'r') as file:
                raw_str = file.read()

            keypoints = []

            x_min = 1000
            y_min = 1000
            x_max = 0
            y_max = 0

            points_str = raw_str.split('\n')[:-1]
            for point_str in points_str:
                _, x_str, y_str = point_str.split(' ')
                x, y = int(float(x_str)), int(float(y_str))

                keypoints.append(x)
                keypoints.append(y)
                keypoints.append(2)

                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # image = cv2.imread(os.path.join(dir_path, image_filename))

            image_field = {}
            image_field["id"] = id_
            image_field["width"] = 640
            image_field["height"] = 480
            image_field["file_name"] = os.path.join(dir_name, image_filename)

            coco_json["images"].append(image_field)

            annotation_field = {}
            annotation_field["keypoints"] = keypoints
            annotation_field["num_keypoints"] = num_keypoints
            annotation_field["id"] = id_
            annotation_field["image_id"] = id_
            annotation_field["category_id"] = 1
            annotation_field["iscrowd"] = 0

            # x_min, y_min = int(np.amin(handpoints[:, 0])), int(np.amin(handpoints[:, 1]))
            # x_max, y_max = int(np.amax(handpoints[:, 0])), int(np.amax(handpoints[:, 1]))

            width, height = x_max - x_min, y_max - y_min
            annotation_field["bbox"] = [x_min, y_min, width, height]
            annotation_field["area"] = width * height

            coco_json["annotations"].append(annotation_field)

            id_ += 1

            # if id_ == 1:
            #     for i in range(21):
            #         cv2.circle(image, (keypoints[i*3+0], keypoints[i*3+1]), 3, (255, 0, 0), 1)
            #         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
            #     plt.imshow(image)
            #     plt.show()
            #     exit()

    return coco_json

if __name__ == "__main__":
    coco_json = raw_to_coco("/home/casey/Downloads/multiview_hand")

    with open("../../data/multiview_hands.json", 'w') as file:
        json.dump(coco_json, file)