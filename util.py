import os
import pandas as pd
from recording import Recording
import random
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import uuid
from sklearn.metrics import classification_report

classes = ['chainsaw', 'clock_tick', 'crackling_fire', 'crying_baby', 'dog', 'helicopter', 'rain', 'rooster',
               'sea_waves', 'sneezing']

def load():
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
                recording_obj = Recording('{0}/{1}'.format(directory, recording))
                class_label = classes.index(metadata[metadata['filename'] == recording]['category'].values[0]) + 1

                recordings.append(recording_obj)
                y.append(class_label)

                # augment recording
                augmentations = recording_obj.augment()
                for augmentation in augmentations:
                    recordings.append(augmentation)
                    y.append(class_label)

    return recordings, y, metadata

def getModelsPath(filename):
    return os.path.join(os.path.abspath('./models'), filename)

def split_dataset(recordings, labels, dir):
    length = len(recordings)
    indices = list(range(0, length))
    random.shuffle(indices)

    train_split_percentage = 0.85
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

    # export test data so that you can use it later for model evaluation
    export_test_data(recordings, labels, test_split_indices, dir)

    return X_train, y_train, X_test, y_test

def export_test_data(recordings, labels, test_split_indices, dir):
    test_dataset_path = os.path.abspath('./test')
    test_recordings = np.take(recordings, test_split_indices, axis=0)
    test_labels = np.take(np.array(labels), test_split_indices, axis=0)
    metadata = []
    for recording, label in zip(test_recordings, test_labels):
        filename = str(uuid.uuid4()) + '.wav'
        recording.export(os.path.join(test_dataset_path, dir, 'audio', filename))
        metadata.append([filename, label])

    metadata = pd.DataFrame(metadata, columns=['filename', 'label'])
    metadata.to_csv(os.path.join(test_dataset_path, dir, 'meta', 'test.csv'), index = False)

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test, verbose=0)

def printResultsPlot(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def load_test(dir):
    recordings = []
    labels = []
    directory = os.path.join(os.path.abspath('./test'), dir, 'audio')
    metadata = pd.read_csv(os.path.join(os.path.abspath('./test'), dir, 'meta', 'test.csv'))
    for recording in sorted(os.listdir(directory)):
        recordings.append(Recording(os.path.join(directory, recording)))
        labels.append(metadata[metadata['filename'] == recording]['label'].values[0])

    if dir == 'model1':
        X_test = np.array(list(map(lambda x: x.mfcc, recordings)))
        print(X_test.shape)
        X_test = X_test.reshape(X_test.shape[0], 431, 13, 1)
        return X_test, np.array(labels)

def test(model, dir):
    model.load_weights(os.path.join(os.path.abspath('./models'), dir + '.hdf5'))
    X_test, y_test = load_test(dir)
    y_pred = model.predict(X_test, batch_size=128, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1) + 1

    print(classification_report(y_test, y_pred_bool))