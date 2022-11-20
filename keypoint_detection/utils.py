import cv2
import glob, os
import numpy as np
import re
import fnmatch
import pickle
import random


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

    # iterate sequences
    for i in range(1,22):
        # read the color frames
        path = dataset_path+"/annotated_frames"+"/data_"+str(i)+"/"
        colorFrames = recursive_glob(path, "*_webcam_[0-9]*")
        colorFrames = natural_sort(colorFrames)
        # print("There are",len(colorFrames),"color frames on the sequence data_"+str(i))
        # read the calibrations for each camera
        # print ("Loading calibration for ./calibrations/data_"+str(i))

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

        for j in range(len(colorFrames)):
            print(colorFrames[j])
            toks1 = colorFrames[j].split("/")
            toks2 = toks1[-1].split("_")
            # print(toks1, toks2)
            # jointPath = toks1[0]+"/"+toks1[1]+"/"+toks1[2]+"/"+toks2[0]+"_joints.txt"
            jointPath = toks1[0]+"/"+toks1[1]+"/"+toks1[2]+"/"+toks1[3]+"/"+toks1[4]+"/"+toks2[0]+"_joints.txt"
            # print(jointPath)
            points3d = readAnnotation3D(jointPath)[0:21] # the last point is the normal

            # project 3d LM points to the image plane
            webcam_id = int(toks2[2].split(".")[0])-1
            print("Calibration for webcam id:",webcam_id)
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

            print(points2d.shape)
            exit()


if __name__ == "__main__":
    load_2D_points("../data/multiview_hand_pose_dataset")