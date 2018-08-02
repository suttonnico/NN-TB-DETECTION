from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D


def cnn():
    raw_model = MobileNet(input_shape=(None, None, 1), include_top=False, weights=None)
    model = Sequential()
    model.add(AveragePooling2D((2, 2), input_shape=img_ds.shape[1:]))
    model.add(BatchNormalization())
    model.add(raw_model)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(disease_vec.shape[1], activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model
