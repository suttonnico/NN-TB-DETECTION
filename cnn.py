from keras import layers, models, optimizers


def cnn(img_width=128, img_height=128):
    # https://www.kaggle.com/crawford/lung-infiltration-cnn-with-keras-on-chest-x-rays
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    # TODO: DROPOUT acá también ponemos?

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64))
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

