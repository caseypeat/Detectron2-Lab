import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random
import json
import matplotlib.pyplot as plt

from tqdm import tqdm


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def recursive_glob(rootdir='.', pattern='*'):
	matches = []
	for root, dirnames, filenames in os.walk(rootdir):
	  for filename in fnmatch.filter(filenames, pattern):
		  matches.append(os.path.join(root, filename))

	return matches

def readAnnotation3D(file):
	f = open(file, "r")
	an = []
	for l in f:
		l = l.split()
		an.append((float(l[1]),float(l[2]), float(l[3])))

	return np.array(an, dtype=float)

def getCameraMatrix():
    Fx = 614.878
    Fy = 615.479
    Cx = 313.219
    Cy = 231.288
    cameraMatrix = np.array([[Fx, 0, Cx],
                    [0, Fy, Cy],
                    [0, 0, 1]])
    return cameraMatrix

def getDistCoeffs():
    return np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])


def load_2D_points(dataset_path):
    cameraMatrix = getCameraMatrix()
    distCoeffs = getDistCoeffs()

    filepath_points_map = {}

    # iterate sequences
    for i in range(1,22):
        # read the color frames
        path = dataset_path+"/annotated_frames"+"/data_"+str(i)+"/"
        colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
        colorFrames = natural_sort(colorFrames)

        # read the calibrations for each camera
        c_0_0 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_1/rvec.pkl","rb"), encoding='latin1')
        c_0_1 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_1/tvec.pkl","rb"), encoding='latin1')
        c_1_0 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_2/rvec.pkl","rb"), encoding='latin1')
        c_1_1 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_2/tvec.pkl","rb"), encoding='latin1')
        c_2_0 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_3/rvec.pkl","rb"), encoding='latin1')
        c_2_1 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_3/tvec.pkl","rb"), encoding='latin1')
        c_3_0 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_4/rvec.pkl","rb"), encoding='latin1')
        c_3_1 = pickle.load(open(dataset_path+"/calibrations/data_"+str(i)+"/webcam_4/tvec.pkl","rb"), encoding='latin1')

        rand_idx = random.randint(0, len(colorFrames))

        # print(colorFrames)

        for j in tqdm(range(len(colorFrames))):
            
            # get joints file path
            toks1 = colorFrames[j].split("/")
            toks2 = toks1[-1].split("_")
            jointPath = ""
            for k in range(len(toks1)-1):
                jointPath += toks1[k]+"/"
            jointPath += toks2[0]+"_joints.txt"

            points3d = readAnnotation3D(jointPath)[0:21] # the last point is the normal

            # project 3d LM points to the image planes
            webcam_id = int(toks2[2].split(".")[0])-1
            if webcam_id == 0:
                rvec = c_0_0
                tvec = c_0_1
            elif webcam_id == 1:
                rvec = c_1_0
                tvec = c_1_1
            elif webcam_id == 2:
                rvec = c_2_0
                tvec = c_2_1
            elif webcam_id == 3:
                rvec = c_3_0
                tvec = c_3_1

            points2d, _ = cv2.projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs)

            filepath_points_map[colorFrames[j]] = points2d[:, 0, :]

            # print(colorFrames[j])

            # image = cv2.imread(colorFrames[j])
            # print(image.shape)
            # exit()
            # for k in range(len(points2d)):
            #     image = cv2.circle(image, (int(points2d[k, 0, 0]), int(points2d[k, 0, 1])), radius=3, color=(0, 0, 255), thickness=-1)
            # plt.imshow(image)
            # plt.show()

            # print(points2d.shape)
            # exit()
        
    return filepath_points_map


def keypoints_to_coco(filepath_points_map, subset_percentage):
    coco_json = {}
    coco_json["images"] = []
    coco_json["annotations"] = []
    coco_json["categories"] = []

    keypoint_names = [
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
    # edges = [
        
    # ]

    category_field = {}
    category_field["keypoints"] = keypoint_names
    # category_field["skeleton"] = []
    category_field["id"] = 1
    category_field["name"] = "hand"
    coco_json["categories"].append(category_field)
    
    id = 1
    for key, val in filepath_points_map.items():
        if random.random() < subset_percentage:
            image_field = {}
            image_field["id"] = id
            image_field["width"] = 640
            image_field["height"] = 480
            image_field["file_name"] = key
            coco_json["images"].append(image_field)

            keypoints = []
            for i in range(len(val)):
                keypoints.append(int(val[i, 0]))
                keypoints.append(int(val[i, 1]))
                keypoints.append(2)

            annotation_field = {}
            annotation_field["keypoints"] = keypoints
            annotation_field["num_keypoints"] = 21
            annotation_field["id"] = id
            annotation_field["image_id"] = id
            annotation_field["category_id"] = 1
            annotation_field["iscrowd"] = 0

            x, y = np.amin(val[:, 0]), np.amin(val[:, 1])
            width, height = np.max(val[:, 0]) - x, np.max(val[:, 1]) - y
            annotation_field["bbox"] = [x, y, width, height]
            annotation_field["area"] = width * height
            coco_json["annotations"].append(annotation_field)


            id += 1

    return coco_json




if __name__ == "__main__":
    filepath_points_map = load_2D_points("/home/casey/Uni/Detectron2-Lab/data/multiview_hand_pose_dataset")

    coco_json = keypoints_to_coco(filepath_points_map, 0.01)

    with open("./labels/hands.json", 'w') as file:
        json.dump(coco_json, file)

    # print(coco_json)