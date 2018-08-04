from keras import layers, models, optimizers
import sklearn
import itertools
import numpy as np
import matplotlib.pylab as plt


def plot_val_acc(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'blue', label='Training acc')
    plt.plot(epochs, val_acc, 'red', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'blue', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def cnn(img_width=128, img_height=128):
    # https://www.kaggle.com/crawford/lung-infiltration-cnn-with-keras-on-chest-x-rays
    layer_C1 = 32
    layer_C2 = 64
    layer_C3 = 128
    dense_layer = 64
    model = models.Sequential()

    model.add(layers.Conv2D(layer_C1, (3, 3), input_shape=(img_width, img_height, 1)))  # the 1 is because greyscale (1 chan)
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    # TODO: DROPOUT acá también ponemos?

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(layer_C2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(layer_C3, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(dense_layer))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("sigmoid"))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        metrics=['acc'])

    model.summary()
    return model

