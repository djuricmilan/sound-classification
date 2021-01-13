import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import librosa
import librosa.display
sb.set(style="white", palette="muted")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from recording import Recording


def add_subplot_axes(ax, position):
    box = ax.get_position()

    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]

    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]])

def plot_recording_overview(recording, ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])

    #display raw_waveforms
    ax_waveform.plot(np.arange(0, len(recording.raw_audio.raw)) / float(Recording.SAMPLING_RATE), recording.raw_audio.raw)
    ax_waveform.get_xaxis().set_visible(False)
    ax_waveform.get_yaxis().set_visible(False)
    ax_waveform.set_title('{0} \n {1}'.format(recording.category, recording.filename), {'fontsize': 8}, y=1.03)

    # display mel_spectograms
    librosa.display.specshow(recording.per_channel_energy_normalized, sr=Recording.SAMPLING_RATE, x_axis='time', y_axis='mel',
                             cmap='RdBu_r')
    ax_spectrogram.get_xaxis().set_visible(False)
    ax_spectrogram.get_yaxis().set_visible(False)

def plot_single_feature_aggregate(feature, title, ax):
    sb.despine()
    ax.set_title(title, y=1.03)
    sb.distplot(feature, bins=20, hist=True, rug=False,
                hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                kde_kws={"shade": False},
                color=sb.color_palette("muted", 4)[1], ax=ax)

if __name__ == '__main__':
    classes = ['chainsaw', 'clock_tick', 'crackling_fire', 'crying_baby', 'dog', 'helicopter', 'rain', 'rooster', 'sea_waves', 'sneezing']

    # set path to dataset
    dataset_path = os.path.abspath('./data' )

    # set path to metadata csv file and audio directories(each class has its own directory)
    metadata_path = os.path.join(dataset_path, 'meta/esc10.csv')
    audio_path = os.path.join(dataset_path, 'audio')

    # Load metadata as a Pandas dataframe
    metadata = pd.read_csv(metadata_path)

    # Examine dataframe's head
    #print(metadata.head())

    # Class distribution
    #print(metadata['category'].value_counts())

    #load entire dataset
    recordings = []
    for directory in sorted(os.listdir('{0}/'.format(audio_path))):
        directory = os.path.join(audio_path, directory)
        if os.path.isdir(directory):
            print('Parsing ' + directory)
            category = []
            for recording in sorted(os.listdir(directory)):
                category.append(Recording('{0}/{1}'.format(directory, recording)))
            recordings.append(category)

    categories = len(classes)

    #seaborn plots
    """
    clips_shown = 5
    f, axes = plt.subplots(nrows=categories, ncols=clips_shown, figsize=(clips_shown * 3, categories * 3), sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.35)
    for c in range(0, categories):
        for i in range(0, clips_shown):
            plot_recording_overview(recordings[c][i], axes[c, i])

    plt.show()
    """

    #distributions of features accros al files belonging to 1 class
    allCatsCoef0 = {}
    allCatsCoef1 = {}
    mfcc_coef_num = 2 #we only look at first 2 mfcc coefs
    for c in range(0, categories):
        aggregateCoef0 = []
        aggregateCoef1 = []
        for i in range(0, len(recordings[c])):
            coef = 0
            aggregateCoef0 = np.concatenate([aggregateCoef0, recordings[c][i].mfcc[:, 0]])

            coef = 1
            aggregateCoef1 = np.concatenate([aggregateCoef1, recordings[c][i].mfcc[:, 1]])

        allCatsCoef0[classes[c]] = aggregateCoef0
        allCatsCoef1[classes[c]] = aggregateCoef1



    #plot_single_feature_aggregate(aggregate, 'Aggregate MFCC_{0} distribution\n(bag-of-frames across all clips\nof {1})'.format(coefficient, clips_50[category][clip].category), ax4)

    fig, ax = plt.subplots(4, 2, sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.8, wspace=0.2)
    for i in range(1, 5):

        ax[i - 1][0].set_title(classes[i - 1] + ' MFC1')
        ax[i - 1][1].set_title(classes[i - 1]  + ' MFC2')

        sb.distplot(allCatsCoef0[classes[i - 1]], bins=20, hist=True, rug=False,
                    hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                    kde_kws={"shade": False},
                    color=sb.color_palette("muted", 4)[1], ax=ax[i - 1][0])

        sb.distplot(allCatsCoef1[classes[i - 1]], bins=20, hist=True, rug=False,
                    hist_kws={"histtype": "stepfilled", "alpha": 0.5},
                    kde_kws={"shade": False},
                    color=sb.color_palette("muted", 4)[1], ax=ax[i - 1][1])

    plt.show()

    #zakljucujemo iz prethodnog plot-a da postoji velika inter-klasna disperzija za vrednosti mfcc1 koeficijenata (mnogo vise nego kod mfcc 2)
    #to je i za ocekivati posto je diskretna kosinusna transformacija koja racuna ove koeficijente vecinu informacija enkapsulirala u nize koeficijente