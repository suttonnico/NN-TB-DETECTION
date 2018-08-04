from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import load_data as ld
import numpy as np
import cnn
import os

dict_characters = {0: 'Normal', 1: 'PTB'}
train = False
# TODO: IDEAS:
# Subir contraste!
# Data augmentation
# PCA o algo


def reshape(array):
    res = np.asarray(array)
    return res.reshape(res.shape + (1,))


def main():
    print("Opening data & labels")
    train_set, train_labels = ld.get_train_set(img_width=128, img_height=128)
    test_set, test_labels = ld.get_test_set(img_width=128, img_height=128)
    test_set = reshape(test_set)
    test_labels = reshape(test_labels)
    train_labels = reshape(train_labels)
    train_set = reshape(train_set)
    print("Open Model")
    epochs = 10
    batch_size = 16
    weights = "weight.hdf5"
    if os.path.exists(weights):
        my_cnn = load_model(weights)
    else:
        my_cnn = cnn.cnn(img_width=128, img_height=128)
    if train:
        # checkpoint
        filepath = "out/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        # train
        history = my_cnn.fit(x=train_set,             # Input should be (train_cases, 128, 128, 1)
                             y=train_labels,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=2,
                             callbacks=callbacks_list,
                             validation_data=(test_set, test_labels)
                             )
        cnn.plot_val_acc(history=history)
    # Confusion Matrix
    Y_pred = my_cnn.predict_classes(test_set)
    # Y_pred_classes = np.argmax(Y_pred, axis=1)
    # import pdb; pdb.set_trace()
    confusion_mtx = confusion_matrix(test_labels, Y_pred)
    cnn.plot_confusion_matrix(confusion_mtx, classes=list(dict_characters.values()))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
