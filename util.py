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

def load_and_augment():
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
                class_label = classes.index(metadata[metadata['filename'] == recording]['category'].values[0])

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


def split_test_train(recordings, labels):
    length = len(recordings)
    indices = list(range(0, length))
    random.shuffle(indices)

    train_split_percentage = 0.85
    split_offset = math.floor(train_split_percentage * length)

    train_split_indices = indices[0:split_offset]
    test_split_indices = indices[split_offset:length]

    # Split recordings with the same indices
    recordings_train = np.take(recordings, train_split_indices, axis=0)
    recordings_test = np.take(recordings, test_split_indices, axis=0)

    # Split labels with the same indices
    labels_train = np.take(np.array(labels), train_split_indices, axis=0)
    labels_test = np.take(np.array(labels), test_split_indices, axis=0)

    # export train data
    export_data(recordings_train, labels_train, 'train')

    # export test data
    export_data(recordings_test, labels_test, 'test')


def export_data(recordings, labels, dir):
    test_dataset_path = os.path.abspath('./augmented')
    metadata = []
    for recording, label in zip(recordings, labels):
        filename = str(uuid.uuid4()) + '.wav'
        recording.export(os.path.join(test_dataset_path, dir, 'audio', filename))
        metadata.append([filename, label])

    metadata = pd.DataFrame(metadata, columns=['filename', 'label'])
    metadata.to_csv(os.path.join(test_dataset_path, dir, 'meta', dir + '.csv'), index = False)

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

def load_augmented(phase, model):
    recordings = []
    labels = []
    directory = os.path.join(os.path.abspath('./augmented'), phase, 'audio')
    metadata = pd.read_csv(os.path.join(os.path.abspath('./augmented'), phase, 'meta', phase + '.csv'))

    for recording in sorted(os.listdir(directory)):
        recordings.append(Recording(os.path.join(directory, recording)))
        labels.append(metadata[metadata['filename'] == recording]['label'].values[0])

    X = []
    if model == 'model1' or model == 'model4':
        X = np.array(list(map(lambda x: x.mfcc, recordings)))
    if model == 'model2' or model == 'model5':
        X = np.array(list(map(lambda x: x.log_amplitude, recordings)))
    if model == 'model3':
        X = np.array(list(map(lambda x: x.raw_audio.raw, recordings)))

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print(X.shape)
    if phase == 'train':
        # one hot encode
        return X, to_categorical(np.array(labels))
    else:
        return X, np.array(labels)

def test(model, model_name):
    model.load_weights(os.path.join(os.path.abspath('./models'), model_name + '.hdf5'))
    X_test, y_test = load_augmented('test', model_name)
    y_pred = model.predict(X_test, batch_size=128, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_bool))

if __name__ == '__main__':
    recordings, y, metadata = load_and_augment()
    split_test_train(recordings, y)