import torch
from torch.utils.data.dataset import Dataset
import sys, pickle
import numpy as np
from glob import glob
from copy import deepcopy
import random


from preprocess import vocab

# create vocab
myvocab = vocab.Vocab()


class Discriminator_dataset(Dataset):
    def __init__(self, real_data_path, fake_data_path,vocab = None):

        self.vocab = myvocab if vocab == None else vocab
        self.event2id = self.vocab.token2id
        self.id2event = self.vocab.id2token

        self.real_data_path = real_data_path
        self.fake_data_path = fake_data_path

        with open(self.real_data_path, 'rb') as file:
            self.real_data = pickle.load(file)
        for i in range(len(self.real_data)):
            self.real_data[i] = [self.real_data[i], 0]

        with open(self.fake_data_path, 'rb') as file:
            self.fake_data = pickle.load(file)
        for i in range(len(self.fake_data)):
            self.fake_data[i] = [self.fake_data[i], 1]

        self.dataset = self.real_data + self.fake_data
        random.shuffle(self.dataset) 

        print('discriminator data loaded')


    def data_pitch_augment(self,src_seq):
        
        all_pitches = [ self.vocab.getPitch(x) for x in src_seq]
        all_pitches = [x for x in all_pitches if x > 0]
        if len(all_pitches) == 0 :
            return src_seq

        pitch_offsets = np.random.randint(1-min(all_pitches),127 - max(all_pitches), size=1)
        pitch_offset = pitch_offsets[0]

        aug_src_phrase = deepcopy(src_seq)
        for t in range(len(aug_src_phrase)):
            if self.vocab.getPitch(aug_src_phrase[t]) > 0:
                aug_src_phrase[t] += pitch_offset
                assert self.vocab.getPitch(aug_src_phrase[t]) > 0

        return aug_src_phrase

    def __getitem__(self, index):

        return np.array(self.data_pitch_augment(self.dataset[index][0])), np.array(self.dataset[index][1])
    
    def __len__(self):
        return len(self.dataset)

class Discriminator_dataset_val(Dataset):
    def __init__(self, real_data_path, fake_data_path,vocab = None):

        self.vocab = myvocab if vocab == None else vocab
        self.event2id = self.vocab.token2id
        self.id2event = self.vocab.id2token

        self.real_data_path = real_data_path
        self.fake_data_path = fake_data_path

        with open(self.real_data_path, 'rb') as file:
            self.real_data = pickle.load(file)
        for i in range(len(self.real_data)):
            self.real_data[i] = [self.real_data[i], 0]

        with open(self.fake_data_path, 'rb') as file:
            self.fake_data = pickle.load(file)
        for i in range(len(self.fake_data)):
            self.fake_data[i] = [self.fake_data[i], 1]

        self.dataset = self.real_data + self.fake_data
        random.shuffle(self.dataset) 

        print('discriminator data loaded')


    def data_pitch_augment(self,src_seq):
        
        all_pitches = [ self.vocab.getPitch(x) for x in src_seq]
        all_pitches = [x for x in all_pitches if x > 0]
        if len(all_pitches) == 0 :
            return src_seq

        pitch_offsets = np.random.randint(1-min(all_pitches),127 - max(all_pitches), size=1)
        pitch_offset = pitch_offsets[0]

        aug_src_phrase = deepcopy(src_seq)
        for t in range(len(aug_src_phrase)):
            if self.vocab.getPitch(aug_src_phrase[t]) > 0:
                aug_src_phrase[t] += pitch_offset
                assert self.vocab.getPitch(aug_src_phrase[t]) > 0

        return aug_src_phrase

    def __getitem__(self, index):

        return np.array(self.dataset[index][0]), np.array(self.dataset[index][1])
    
    def __len__(self):
        return len(self.dataset)
    
    

dataset = Discriminator_dataset(real_data_path='data_pkl/discriminator_realdata.pkl',
                                fake_data_path='data_pkl/discriminator_fakedata.pkl')