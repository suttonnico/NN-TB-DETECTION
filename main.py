from keras.callbacks import ModelCheckpoint
import load_data as ld
import numpy as np
import cnn

# TODO: IDEAS:
# Subir contraste!
# Data augmentation
# PCA o algo


def main():
    print("Opening data & labels")
    train_set, train_labels = ld.get_train_set(img_width=128, img_height=128)
    test_set, test_labels = ld.get_test_set(img_width=128, img_height=128)
    test_set = np.asarray(test_set)
    test_labels = np.asarray(test_labels)
    train_labels = np.asarray(train_labels)
    train_set = np.asarray(train_set)
    # import pdb; pdb.set_trace()
    print("Trining Net")
    epochs = 10
    batch_size = 16
    my_cnn = cnn.cnn(img_width=128, img_height=128)
    # checkpoint
    filepath = "out/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # train
    my_cnn.fit(x=train_set,             # Input should be (train_cases, 128, 128, 1)
               y=train_labels,
               batch_size=batch_size,
               epochs=epochs,
               verbose=2,
               callbacks=callbacks_list,
               validation_data=(test_set, test_labels)
               )
    print("Done Training...")


if __name__ == "__main__":
    main()
