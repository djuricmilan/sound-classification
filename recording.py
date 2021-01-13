import librosa
import pydub
import numpy as np
import os
import glob
import random

class Recording:

    #frekvencija uzorkovanja(sampling rate)
    SAMPLING_RATE = 44100

    FRAME = 512

    class RawAudio:

        def __init__(self, absolute_file_path):
            self.absolute_file_path = absolute_file_path
            self.load()

        def load(self):
            # Actual recordings are sometimes not frame accurate
            # we create a silent audio segment of 5 seconds, and then overlay it with the signal from the input file
            self.data = pydub.AudioSegment\
                .silent(duration=5000)\
                .overlay(pydub.AudioSegment.from_file(self.absolute_file_path)[0:5000])

            #convert to float and normalize to interval [-1, 1]]
            self.raw = (np.frombuffer(self.data._data, dtype="int16") + 0.5) / (32767 + 0.5)  # convert to float
            #print(self.raw)
            #return (self)

        #def __exit__(self, exception_type, exception_value, traceback):
        #    if exception_type is not None:
        #        print
        #        exception_type, exception_value, traceback
        #    del self.data
        #    del self.raw

    def __init__(self, filename):
        self._init_fs_vars(filename)
        self.category = self.directory_name
        self.raw_audio = Recording.RawAudio(self.absolute_file_path)
        self._calculate_mfcc()

    def _init_fs_vars(self, filename):
        self.filename = os.path.basename(filename)
        self.absolute_file_path = filename
        self.directory_name = os.path.basename(os.path.dirname(self.absolute_file_path))

    def _calculate_mfcc(self):
        # MFCC computation
        # hop_length - broj pomeraja pri semplovanju sledeceg frejma. U ovom slucaju isto koliko i duzina frejma (nema preklapanja frejmova)
        # window je velicine 2048
        # 128 bands
        self.mel_spectrogram = librosa.feature.melspectrogram(self.raw_audio.raw,
                                                              win_length=2048,
                                                              sr=Recording.SAMPLING_RATE,
                                                              hop_length=Recording.FRAME)

        #input 1: log-scaling of mel_spectogram
        self.log_amplitude = librosa.amplitude_to_db(self.mel_spectrogram)

        #mfcc shape = (431, 13); 431 = (audioLenght in seconds * sampling rate / hop_length) = (5*41000)/512
        # input 2: mfcc of mel_spectogram
        self.mfcc = librosa.feature.mfcc(S=self.log_amplitude, n_mfcc=13).transpose()

        #http://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf
        # input 3: per channel enery normalized mel-spectogram
        self.per_channel_energy_normalized = librosa.pcen(self.mel_spectrogram)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Recording.FRAME):(index + 1) * Recording.FRAME]

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)

if __name__ == '__main__':
    filepath = os.path.abspath('data/audio/dog/1-30226-A-0.wav')
    recording = Recording(filepath)
