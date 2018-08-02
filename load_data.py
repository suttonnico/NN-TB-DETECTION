import numpy as np
import os
import cv2
from sklearn import decomposition


def get_labels(labels_dir='data/ChinaSet_AllFiles/ClinicalReadings/', n=662):
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
    return labels


def get_images(imgs_dir='data/ChinaSet_AllFiles/CXR_png/'):
    extensions = {".jpg", ".png", ".gif"}  # etc
    # file: any(file.endswith(ext) for ext in extensions) for file in files
    i = 0
    # make sure the file is a image
    imgs_files = [f for f in os.listdir(imgs_dir) if any(f.endswith(ext) for ext in extensions)]
    images = np.zeros((len(imgs_files), 290, 300))
    for f in imgs_files:
        # print(f)
        img = cv2.imread(os.path.join(imgs_dir, f), cv2.IMREAD_GRAYSCALE)        # img.shape ~ (2919, 3000)
        # print(img.shape)
        img = cv2.resize(img, (300, 290))
        images[i] = img     # Reshape images TODO: Elegir éste valor de resize acorde
        i += 1
    return images
