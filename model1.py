"""
    First model uses log mel spectogram images as input to the network
"""
from util import load, split_dataset, getModelsPath
from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, ReLU, Softmax
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from datetime import datetime
import os

num_rows = 431
num_columns = 13
num_channels = 1
num_labels = 10
num_epochs = 250
batch_size = 128


def create_model():
    # Create a secquential object
    model = Sequential()

    # Conv layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(num_rows, num_columns, num_channels)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())

    # Softmax output
    model.add(Dense(num_labels))
    model.add(Softmax())

    return model

if __name__ == '__main__':
    #load dataset
    """
        librosa.load does the necessary preprocessing for us
        1. converts sampling rate to 22.05 kHz
        2. normalizes bit depth range to (-1,1)
        3. flattens the audio channel to mono
    """
    recordings, labels, metadata = load()

    #split dataset for training and
    X_train, y_train, X_test, y_test = split_dataset(recordings, labels)

    #reshape dataset to ensure channel last format (samples, height, width, channels)
    X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
    X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

    #create model
    model = create_model()
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=1e-4, beta_1=0.99, beta_2=0.999))
    model.summary()

    # Save checkpoints
    model_file = 'model1.hdf5'
    model_path = getModelsPath(model_file)
    bestModelCheckpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')

    #train network
    start = datetime.now()
    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_split=0.1,
                        callbacks=[bestModelCheckpoint],
                        verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
