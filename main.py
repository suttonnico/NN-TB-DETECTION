import load_data as ld
import cv2
import random
# import cnn

# TODO: IDEAS:
# Subir contraste!
# Data augmentation
# PCA o algo


def main():
    print("Opening data & labels")
    labels = ld.get_labels()
    train_set = ld.get_train_set(img_width=300, img_height=300)
    test_set = ld.get_test_set(img_width=300, img_height=300)
    import pdb; pdb.set_trace()
    print("Trining Net")
    # my_cnn = cnn.cnn(img_width=300, img_height=300)


if __name__ == "__main__":
    main()
