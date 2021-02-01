import os
import pandas as pd
from recording import Recording
import random
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def load():
    classes = ['chainsaw', 'clock_tick', 'crackling_fire', 'crying_baby', 'dog', 'helicopter', 'rain', 'rooster',
               'sea_waves', 'sneezing']

    # set path to dataset
    dataset_path = os.path.abspath('./data')

    # set path to metadata csv file and audio directories(each class has its own directory)
    metadata_path = os.path.join(dataset_path, 'meta/esc10.csv')
    audio_path = os.path.join(dataset_path, 'audio')

    # load metadata
    metadata = pd.read_csv(metadata_path)

    # load entire dataset
    recordings = []
    y = []
    for directory in sorted(os.listdir('{0}/'.format(audio_path))):
        directory = os.path.join(audio_path, directory)
        if os.path.isdir(directory):
            print('Parsing ' + directory)
            for recording in sorted(os.listdir(directory)):
                recordings.append(Recording('{0}/{1}'.format(directory, recording)))
                y.append(classes.index(metadata[metadata['filename'] == recording]['category'].values[0]) + 1)

    return recordings, y, metadata

def getModelsPath(filename):
    return os.path.join(os.path.abspath('./models'), filename)

def split_dataset(recordings, labels):
    length = len(recordings)
    indices = list(range(0, length))
    random.shuffle(indices)

    train_split_percentage = 0.8
    split_offset = math.floor(train_split_percentage * length)

    train_split_indices = indices[0:split_offset]
    test_split_indices = indices[split_offset:length]

    # Split the features
    mfccs = list(map(lambda recording: recording.mfcc, recordings))
    X_train = np.take(mfccs, train_split_indices, axis=0)
    X_test = np.take(mfccs, test_split_indices, axis=0)

    # Split labels with the same indices
    y_train = np.take(np.array(labels), train_split_indices, axis=0)
    y_test = np.take(np.array(labels), test_split_indices, axis=0)

    # Print status
    print("X test shape: {} \t X train shape: {}".format(X_test.shape, X_train.shape))
    print("y test shape: {} \t\t y train shape: {}".format(y_test.shape, y_train.shape))

    #one hot encode labels
    le = LabelEncoder()
    y_train = to_categorical(le.fit_transform(y_train))
    y_test = to_categorical(le.fit_transform(y_test))

    return X_train, y_train, X_test, y_test