import load_data as ld
import cv2


def main():
    labels = ld.get_labels()
    img = ld.get_images()
    cv2.imshow(str(labels[0]), img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
