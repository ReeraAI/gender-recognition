import os
import librosa
import numpy as np

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)

    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    
    return result


if __name__ == "__main__":

    from glob import glob
    from random import randint
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Gender recognition script, this will load the model you trained, and perform inference on a sample you provide.")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()
    file = args.file

    if not file or not os.path.isfile(file):
        if input("Failed to use file, should random sound be used? Yes/No (Default: Yes)\n=> ").lower() == 'no':
            exit()
        
        # if file not provided, or it doesn't exist, use random voice
        files = glob(r"samples/*.wav")
        file = files[randint(0, len(files) - 1)]
        print("Voice:", file)
    
    from utils import load_data, split_data, create_model

    # construct the model
    model = create_model()

    # load the saved/trained weights
    model.load_weights("results/model.h5")

    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)

    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"

    # show the result!
    print("Result:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
