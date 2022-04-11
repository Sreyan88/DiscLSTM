import pandas as pd
import glob

def construct(d,u):
    return f'dia{d}_utt{u}.wav'

df = pd.read_csv('test_sent_emo.csv')
csv_files = []

for i,row in df.iterrows():
    csv_files.append(construct(row['Dialogue_ID'],row['Utterance_ID']))

import os
os.chdir('test_splits_audio/')
system_files = sorted(glob.glob('*'))
os.chdir('../')

x = [ele for ele in csv_files if ele not in system_files]
y = [ele for ele in system_files if ele not in csv_files]
print(x)
print(y)