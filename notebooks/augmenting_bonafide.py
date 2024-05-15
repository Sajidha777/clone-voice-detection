import librosa
import soundfile as sf
from audiomentations import Compose, TimeMask, TimeStretch, RoomSimulator, Gain
import os

import pandas as pd

AG1 = Compose([
    TimeMask(min_band_part=0.1, max_band_part=0.15,fade=True,p=1.0),
    TimeStretch(min_rate=0.7, max_rate=0.9, p=0.5),
])

AG2 = Compose([
    TimeStretch(min_rate=1.05, max_rate=1.25, p=1),
    Gain(p=0.5),
])

AG3 = Compose([
    RoomSimulator(p=1.0),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    Gain(p=0.5),
])



if __name__=="__main__":
    
    df = pd.read_csv('/home/saji/DSprojects/CloneVoiceDetection/data/reduced_LA/modified_1500.csv')

    all_bonafide = df.loc[df['target'] == 'bonafide', 'audio_ID']
    del df
    ip_folder_path = "/home/saji/DSprojects/CloneVoiceDetection/data/reduced_LA/flac/"
    op_folder_path = "/home/saji/DSprojects/CloneVoiceDetection/data/reduced_LA/flac_augmented_bonafide/"

    for filename in (all_bonafide):
        filepath = os.path.join(ip_folder_path,filename+'.flac')
        signal, sr = librosa.load(filepath)
        
        augmented1 = AG1(signal, sr)
        sf.write(op_folder_path+filename+'AG1'+'.flac', augmented1, sr)
        
        augmented2 = AG2(signal, sr)
        sf.write(op_folder_path+filename+'AG2'+'.flac', augmented2, sr)
        
        augmented3 = AG3(signal, sr)
        sf.write(op_folder_path+filename+'AG3'+'.flac', augmented3, sr)
        
