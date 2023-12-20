from tkinter import filedialog
import librosa
import numpy as np
import tensorflow
import os
from pydub import AudioSegment


SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
MODEL = tensorflow.keras.models.load_model("A_Model_KFold.h5")


def predict(model, X):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Predicted label: {}".format(predicted_index))
    return predicted_index


def audio_preparation(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T
        mfcc = mfcc[..., np.newaxis]
    return mfcc


def music_genre(number):
    if number == 0:
        print("Music Genre: Blue")

    elif number == 1:
        print("Music Genre: Classic")

    elif number == 2:
        print("Music Genre: Country")

    elif number == 3:
        print("Music Genre: Disco")

    elif number == 4:
        print("Music Genre: Hiphop")

    elif number == 5:
        print("Music Genre: Jazz")

    elif number == 6:
        print("Music Genre: Metal")

    elif number == 7:
        print("Music Genre: Pop")

    elif number == 8:
        print("Music Genre: Reggae")

    elif number == 9:
        print("Music Genre: Rock")


if __name__ == "__main__":

    file_path = filedialog.askopenfilename(initialdir="/", title="Select Music File", filetypes = (('MP3 files','*.mp3'),('WAV files','*.wav')))
    file_extension = os.path.splitext(file_path)
    file_extension = file_extension[1]

    if file_extension == ".wav":
        mfcc_value = audio_preparation(file_path)

        print("Input File For Model: {}".format(file_path))
        Index = predict(MODEL, mfcc_value)
        music_genre(Index)

    elif file_extension == ".mp3":
        src = file_path
        dst = "test.wav"

        # convert wav to mp3
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")

        mfcc_value = audio_preparation("test.wav")

        print("Input File For Model: {}".format(file_path))
        Index = predict(MODEL, mfcc_value)
        music_genre(Index)

    else:
        print("Error! Wrong File Type Selected")
