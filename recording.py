import librosa
import pydub
import numpy as np
import os
import glob
import random
from pydub.playback import play
from datetime import datetime
from random import randint

class Recording:

    #frekvencija uzorkovanja(sampling rate)
    SAMPLING_RATE = 44100

    FRAME = 512

    class RawAudio:

        def __init__(self, absolute_file_path = ''):
            if absolute_file_path != '':
                self.absolute_file_path = absolute_file_path
                self.load()

        def load(self):
            # Actual recordings are sometimes not frame accurate
            # we create a silent audio segment of 5 seconds, and then overlay it with the signal from the input file
            self.data = pydub.AudioSegment\
                .silent(duration=5000)\
                .overlay(pydub.AudioSegment.from_file(self.absolute_file_path, format="wav")[0:5000])

            #convert to float and normalize to interval [-1, 1]]
            self.raw = (np.frombuffer(self.data._data, dtype="int16") + 0.5) / (32767 + 0.5)  # convert to float
            #print(self.raw)
            #return (self)

        def load_from_audiosegment(self, audiosegment):
            self.data = audiosegment
            self.raw = (np.frombuffer(self.data._data, dtype="int16") + 0.5) / (32767 + 0.5)  # convert to float

    def __init__(self, filename = ''):
        if(filename != ''):
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
        # dimension (128, 431)
        self.log_amplitude = librosa.amplitude_to_db(self.mel_spectrogram)

        #mfcc shape = (431, 13); 431 = (audioLenght in seconds * sampling rate / hop_length) = (5*41000)/512
        # input 2: mfcc of mel_spectogram
        self.mfcc = librosa.feature.mfcc(S=self.log_amplitude, n_mfcc=13).transpose()

        #http://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf
        # input 3: per channel energy normalized mel-spectogram
        self.per_channel_energy_normalized = librosa.pcen(self.mel_spectrogram)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Recording.FRAME):(index + 1) * Recording.FRAME]

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)

    def play_audio(self):
        start = datetime.now()
        play(self.raw_audio.data)
        end = datetime.now()
        print(end - start)

    def augment(self):
        #self.play_audio()
        augmentations = []
        for i in range(5):
            fade_in = random.randint(1, 5000)
            #print("Fade in: " + str(fade_in))
            fade_out = 5000 - fade_in
            augmented_audio = self.augment_fade(fade_in, fade_out)
            #augmented_audio.play_audio()
            augmentations.append(augmented_audio)

        for i in range(5):
            fade_in = random.randint(1, 5000)
            #print("Fade in: " + str(fade_in))
            fade_out = 5000 - fade_in
            augmented_audio = self.augment_fade_reverse(fade_in, fade_out)
            #augmented_audio.play_audio()
            augmentations.append(augmented_audio)

        for i in range(10):
            factor_beginning = random.randint(5, 10)
            factor_end = random.randint(5, 10)

            beginning_multiplier = random.randint(0,1)
            if(beginning_multiplier == 0):
                factor_beginning *= -1
            end_multiplier = random.randint(0, 1)
            if (end_multiplier == 0):
                factor_end *= -1

            #print("Factor beginning: " + str(factor_beginning))
            #print("Factor end: " + str(factor_end))

            augmented_audio = self.augment_change_value(factor_beginning=factor_beginning, factor_end=factor_end)
            #augmented_audio.play_audio()
            augmentations.append(augmented_audio)

        augmented_audio = self.change_pitch(octaves=-0.25)
        #augmented_audio.play_audio()
        augmentations.append(augmented_audio)
        augmented_audio = self.change_pitch(octaves=-0.5)
        #augmented_audio.play_audio()
        augmentations.append(augmented_audio)
        augmented_audio = self.change_pitch(octaves=-0.75)
        #augmented_audio.play_audio()
        augmentations.append(augmented_audio)
        augmented_audio = self.change_pitch(octaves=-0.1)
       # augmented_audio.play_audio()
        augmentations.append(augmented_audio)

        return augmentations

    def augment_change_value (self, border=2501, factor_beginning = 5, factor_end = 5):
        beginning = self.raw_audio.data[:border]
        beginning += factor_beginning
        end = self.raw_audio.data[border:]
        end -= factor_end
        new_value_sound = beginning + end
        recording = Recording()
        recording.raw_audio = Recording.RawAudio()
        recording.raw_audio.load_from_audiosegment(new_value_sound)
        recording._calculate_mfcc()
        return recording

    def augment_fade(self, fade_in=2000, fade_out=3000):
        faded_sound = self.raw_audio.data.fade_in(fade_in).fade_out(fade_out)
        recording = Recording()
        recording.raw_audio = Recording.RawAudio()
        recording.raw_audio.load_from_audiosegment(faded_sound)
        recording._calculate_mfcc()
        return recording

    def augment_fade_reverse(self, fade_in=2000, fade_out=3000):
        faded_sound = self.raw_audio.data.fade_out(fade_out).fade_in(fade_in)
        recording = Recording()
        recording.raw_audio = Recording.RawAudio()
        recording.raw_audio.load_from_audiosegment(faded_sound)
        recording._calculate_mfcc()
        return recording

    def change_pitch(self, octaves = -0.5):
        # shift the pitch down by half an octave (speed will decrease proportionally)
        new_sample_rate = int(self.raw_audio.data.frame_rate * (2.0 ** octaves))
        lowpitch_sound = self.raw_audio.data._spawn(self.raw_audio.data.raw_data, overrides={'frame_rate': new_sample_rate})
        lowpitch_sound = lowpitch_sound.set_frame_rate(Recording.SAMPLING_RATE)

        length = self.raw_audio.data.duration_seconds * 1000
        new_length = lowpitch_sound.duration_seconds * 1000
        lowpitch_sound = lowpitch_sound[int((new_length - length) / 2): int((new_length + length) / 2)]

        recording = Recording()
        recording.raw_audio = Recording.RawAudio()
        recording.raw_audio.load_from_audiosegment(lowpitch_sound)
        recording._calculate_mfcc()
        return recording

    def export(self, filepath):
        self.raw_audio.data.export(filepath, format="wav")

if __name__ == '__main__':
    filepath = os.path.abspath('data/audio/dog/1-30226-A-0.wav')
    recording = Recording(filepath)
    recording.augment()