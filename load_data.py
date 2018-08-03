import numpy as np
import os
import cv2
import pickle
from random import uniform


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


def save_images(imgs_dir='data/ChinaSet_AllFiles/CXR_png/', img_width=128, img_height=128):
    extensions = {".jpg", ".png", ".gif"}  # etc
    # make sure the file is a image
    imgs_files = [f for f in os.listdir(imgs_dir) if any(f.endswith(ext) for ext in extensions)]
    images = []
    for f in imgs_files:
        # print(f)
        img = cv2.imread(os.path.join(imgs_dir, f), cv2.IMREAD_GRAYSCALE)        # img.shape ~ (2919, 3000)
        img = cv2.resize(img, (img_width, img_height))
        images.append(img)     # Reshape images TODO: Elegir éste valor de resize acorde
    with open("data/images", 'wb') as fp:
        pickle.dump(images, fp)


def get_labels():
    if not os.path.exists("data/labels"):
        save_labels()
    with open("data/labels", 'rb') as fp:
        return pickle.load(fp)


def get_images(imgs_dir='data/ChinaSet_AllFiles/CXR_png/', img_width=128, img_height=128):
    if not os.path.exists("data/images"):
        save_images(imgs_dir, img_width, img_height)
    with open("data/images", 'rb') as fp:
        return pickle.load(fp)


def get_train_set(img_width=128, img_height=128):
    if not os.path.exists("data/train"):
        generate_train_set(img_width, img_height)
    with open("data/train", 'rb') as fp:
        train_set = pickle.load(fp)
    with open("data/train_labels", 'rb') as fp:
        train_labels = pickle.load(fp)
    return train_set, train_labels


def get_test_set(img_width=128, img_height=128):
    if not os.path.exists("data/test"):
        generate_train_set(img_width, img_height)
    with open("data/test", 'rb') as fp:
        test_set = pickle.load(fp)
    with open("data/test_labels", 'rb') as fp:
        test_labels = pickle.load(fp)
    return test_set, test_labels


def generate_train_set(img_width=128, img_height=128):
    images = get_images(img_width=img_width, img_height=img_height)
    labels = get_labels()
    train_set_percentage = 0.8  # Cuanto porcentaje de las imágenes uso para el train set.
    # normal_cases = [img for i, img in enumerate(images) if labels[i] == 0]
    # ptb_cases = [img for i, img in enumerate(images) if labels[i] == 1]
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    for i, c in enumerate(images):
        if uniform(0, 1) > train_set_percentage:
            test_set.append(c)
            test_labels.append(labels[i])
        else:
            train_set.append(c)
            train_labels.append(labels[i])
    with open("data/train", 'wb') as fp:
        pickle.dump(train_set, fp)
    with open("data/test", 'wb') as fp:
        pickle.dump(test_set, fp)
    with open("data/train_labels", 'wb') as fp:
        pickle.dump(train_labels, fp)
    with open("data/test_labels", 'wb') as fp:
        pickle.dump(test_labels, fp)
