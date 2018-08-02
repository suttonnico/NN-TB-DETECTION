import load_data as ld
import cv2
# import cnn

# TODO: IDEAS:
# Subir contraste!
# Data augmentation
# PCA o algo


def main():
    print("Opening data & labels")
    labels = ld.get_labels()
    images = ld.get_images(img_width=300, img_height=300)
    # cv2.imshow(str(labels[0]), images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    train_set_percentage = 0.8  # Cuanto porcentaje de las imágenes uso para el train set.
    normal_cases = [img for i, img in enumerate(images) if labels[i] == 0]
    ptb_cases = [img for i, img in enumerate(images) if labels[i] == 1]
    import pdb; pdb.set_trace()
    print("Trining Net")
    # my_cnn = cnn.cnn(img_width=300, img_height=300)


if __name__ == "__main__":
    main()
