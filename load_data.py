import numpy as np
import os
import cv2
import pickle


def save_labels(labels_dir='data/ChinaSet_AllFiles/ClinicalReadings/', n=662):
    labels = np.zeros(n)
    labels_files = [f for f in os.listdir(labels_dir)]
    for f in labels_files:
        flag = 0
        with open(labels_dir+'/'+f) as f_txt:
            content = f_txt.readlines()
            content_lines = [x.strip() for x in content]        # Leo las anotaciones
            for c in content_lines:
                if c.find('normal') != -1:                      # Si es normal
                    labels[int(f[7:11])-1] = 0                  # label = 0 -> Normal
                    flag = 1                                    # flag = 1
            if flag == 0:
                labels[int(f[7:11])-1] = 1                      # label = 1 -> PTB
    with open("data/labels", 'wb') as fp:
        pickle.dump(labels, fp)


def save_images(imgs_dir='data/ChinaSet_AllFiles/CXR_png/'):
    extensions = {".jpg", ".png", ".gif"}  # etc
    # file: any(file.endswith(ext) for ext in extensions) for file in files
    i = 0
    # make sure the file is a image
    imgs_files = [f for f in os.listdir(imgs_dir) if any(f.endswith(ext) for ext in extensions)]
    images = []
    for f in imgs_files:
        # print(f)
        img = cv2.imread(os.path.join(imgs_dir, f), cv2.IMREAD_GRAYSCALE)        # img.shape ~ (2919, 3000)
        img = cv2.resize(img, (300, 300))
        images.append(img)     # Reshape images TODO: Elegir éste valor de resize acorde
    with open("data/images", 'wb') as fp:
        pickle.dump(images, fp)


def get_labels():
    if not os.path.exists("data/labels"):
        save_labels()
    with open("data/labels", 'rb') as fp:
        return pickle.load(fp)


def get_images():
    if not os.path.exists("data/images"):
        save_images()
    with open("data/images", 'rb') as fp:
        return pickle.load(fp)
