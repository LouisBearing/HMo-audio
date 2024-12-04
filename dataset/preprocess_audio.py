import numpy as np
import pickle
import os
from scipy.io import wavfile
import python_speech_features
from argparse import ArgumentParser


def get_audio_feature_from_audio(audio_dir, file, style=1):
    '''
    From https://github.com/FuxiVirtualHuman/AAAI22-one-shot-talking-face
    '''
    audio_path = os.path.join(audio_dir, file)
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio) == 0:
        return np.array([]), np.array([])

    if sample_rate != 16000:
        temp_audio = os.path.join(audio_dir, "temp.wav")
        command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio)
        os.system(command)
        sample_rate, audio = wavfile.read(temp_audio)
        os.remove(temp_audio)
    
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]

    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    if style == 1:
        mfcc = python_speech_features.mfcc(audio, sample_rate)
        melbank = python_speech_features.logfbank(audio, sample_rate, nfilt=26)
    elif style == 2:
        mfcc = 0
        melbank = python_speech_features.logfbank(audio, sample_rate, nfilt=80)
    elif style == 3:
        mfcc = 0
        melbank = python_speech_features.ssc(audio)    
    
    return mfcc, melbank

def process_audio_dir(args):

    audio_dir = args.audio_dir
    style = args.prepro_style
    add = str(style) if style > 1 else ''
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    for i, file in enumerate(audio_files):
        f_idx = file.split('.')[0]
        out_path = os.path.join(audio_dir, f_idx + '_audiofeats' + add)
        if os.path.exists(out_path):
            continue
        mfcc, melbank = get_audio_feature_from_audio(audio_dir, file, style)
        if style == 1:
            if len(mfcc) == 0:
                continue
            with open(out_path, 'wb') as f:
                pickle.dump(np.concatenate([mfcc, melbank], axis=1), f)
        elif (style == 2) or (style == 3):
            with open(out_path, 'wb') as f:
                pickle.dump(melbank, f)
        if i % 50 == 0:
            print(i)

def main(args):
    audio_dir = args.audio_dir
    if os.path.exists(os.path.join(audio_dir, "temp.wav")):
        os.remove(os.path.join(audio_dir, "temp.wav"))
    process_audio_dir(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", help='Path to wav files')
    parser.add_argument("--prepro_style", help='1: one-shot style, 13 mfcc + 26 fbank, 2: 80 fbanks, 3: 26 subband spectral coeffs', type=int)
    args = parser.parse_args()
    print('Arguments received, calling main !')
    main(args)