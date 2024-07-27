import vocab
import numpy as np
import glob
import os
import pickle
from tqdm import tqdm

myvocab = vocab.Vocab()


# Check the paths for your own case
# the theme annotated midi files
MIDI_FILES = "./POP909_Pure_Train/*.mid"

# the output training data 
OUTPUT_DATA_PKL = "/data_pkl/discriminator_fake_data.pkl"

all_mids = sorted(glob.glob(MIDI_FILES))

fixed_length = 512
padding_value = 0
dataset = []


def slice_sequence(seq, fixed_length):
    sliced_sequence = [seq[i:i+fixed_length] for i in range(0, len(seq), fixed_length)]
    return sliced_sequence

def pad_sequence(seq, fixed_length, pad_value):
    padded_sequence = seq + [pad_value] * (fixed_length - len(seq))
    return padded_sequence



for _midiFile in tqdm(all_mids):
    # convert midi files to token representation and save as .pkl file
    output_pkl_fp = _midiFile.replace(".mid",".pkl")
    remi_seq = myvocab.midi2REMI(_midiFile,include_bridge=False,bar_first=False,verbose=False)
    ret = remi_seq

    if len(ret) > 512:
        sliced_sequence = slice_sequence(ret, fixed_length=fixed_length)
        sliced_sequence[-1] = pad_sequence(sliced_sequence[-1], fixed_length, padding_value)
        for i in range(len(sliced_sequence)):
            dataset.append(sliced_sequence[i])
    else:
        sliced_sequence = pad_sequence(ret, fixed_length, padding_value)
        dataset.append(sliced_sequence)
        


with open('Discriminator_Real_Data.pkl', 'wb') as file:
    pickle.dump(dataset, file)