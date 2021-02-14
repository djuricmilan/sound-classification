"""
    Like the second model uses log mel spectogram images as input to the network
    Spatial dropout and dense layers instead of average pooling, leaky ReLU
"""
from util import load_augmented, getModelsPath, evaluate_model, printResultsPlot, test
from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, SpatialDropout2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Softmax, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import os
import sys

num_rows = 40
num_columns = 431
num_channels = 1
num_labels = 10
num_epochs = 50
batch_size = 128
model_name = 'model5'

def create_model():
    # Create a secquential object
    model = Sequential()

    # Conv layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(num_rows, num_columns, num_channels)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(SpatialDropout2D(0.1)) # drops entire 2D feature maps instead of individual elements

    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    # classifier
    model.add(Flatten())

    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.3))

    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.3))

    # Softmax output
    model.add(Dense(num_labels))
    model.add(Softmax())

    return model


def train(model):
    # load dataset
    X_train, y_train = load_augmented('train', model_name)

    # Callbacks
    model_file = model_name + '.hdf5'
    model_path = getModelsPath(model_file)
    bestModelCheckpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', mode='min')

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, mode='min', min_lr=1e-5)

    # train network
    start = datetime.now()
    history = model.fit(X_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        validation_split=0.2,
                        callbacks=[bestModelCheckpoint, reduce_lr_loss],
                        verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # print results
    printResultsPlot(history)

    # evaluate model
    model = load_model(model_path)
    X_test, y_test = load_augmented('test', model_name)
    test_results = evaluate_model(model, X_test, y_test)
    print("Test accuracy: " + str(test_results[1]))
    print("Test loss: " + str(test_results[0]))

if __name__ == '__main__':
    # create model
    model = create_model()
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=1e-4, beta_1=0.99, beta_2=0.999))
    model.summary()

    if sys.argv[1] == "train":
        train(model)
    else:
        test(model, model_name)