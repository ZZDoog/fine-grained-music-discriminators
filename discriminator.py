"""Rhythm Transformer Disciminator

A Transformer Encoder Discriminator for Symbolic Music
Revised from pytorch transformer implementation

    Author: ZZDoog
    Email: zhedong_zhang@hdu.edu.cn
    Date: 2023.5.20
    
"""
import torch
import copy
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from myTransformer import TransformerEncoder, TransformerEncoderLayer
from mymodel import PositionalEncoding, WordEmbedding


def Pitch_discriminator_infill(matrix):
    batch_size, seq_len = matrix.size()

    for i in range(batch_size):
        j = 0
        while j < seq_len:
            if matrix[i, j] != 0:
                if j < seq_len - 1:
                    matrix[i, j+1] = matrix[i, j]
                if j < seq_len - 2:
                    matrix[i, j+2] = matrix[i, j]
                j += 3
            else:
                j += 1

    return matrix




class FocalLoss(Module):


    def __init__(self, alpha=1, gamma=2, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, input, target):

        log_probs = F.log_softmax(input, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, target.view(-1, 1)).squeeze(1)
        loss = -self.alpha * (1 - pt) ** self.gamma * log_probs.gather(1, target.view(-1, 1)).squeeze(1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss





class TransformerDiscriminator(Module):

    def __init__(self, ntoken, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        
        super(TransformerDiscriminator, self).__init__()

        self.ntoken = ntoken
        self.d_model = d_model
        self.nhead = nhead

        self.pos_enc = PositionalEncoding(self.d_model,dropout=dropout)

        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder_norm = LayerNorm(d_model)

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, self.encoder_norm)
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)

        self.fc1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.fc2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, int(self.d_model/4))
        self.fc4 = torch.nn.Linear(int(self.d_model/4), 2)


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        
        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    
    def discriminate_forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = torch.matmul(src, self.token_embedding.weight)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info

    

    def get_embedding(self, src, src_mask=None, src_key_padding_mask=None):

        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        return info

    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                # torch.nn.init.kaiming_uniform_(p)
                # torch.nn.init.normal_(p,mean=0,std=0.01)
                xavier_uniform_(p)




class Pitch_Discriminator(Module):

    def __init__(self, vocab, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        
        super(Pitch_Discriminator, self).__init__()

        self.vocab = vocab
        self.ntoken = self.vocab.n_tokens


        self.NOTE_MELODY_lower_bound = self.vocab.token2id['Note-On-MELODY_1']
        self.NOTE_MELODY_upper_bound = self.vocab.token2id['Note-On-MELODY_127']

        self.NOTE_BRIDGE_lower_bound = self.vocab.token2id['Note-On-BRIDGE_1']
        self.NOTE_BRIDGE_upper_bound = self.vocab.token2id['Note-On-BRIDGE_127']

        self.NOTE_PIANO_lower_bound = self.vocab.token2id['Note-On-PIANO_1']
        self.NOTE_PIANO_upper_bound = self.vocab.token2id['Note-On-PIANO_127']

        self.NOTE_Velocity_lower_bound = self.vocab.token2id['Note-Velocity-MELODY_1']
        self.NOTE_Velocity_upper_bound = self.vocab.token2id['Note-Velocity-PIANO_126']


        self.d_model = d_model
        self.nhead = nhead

        self.pos_enc = PositionalEncoding(self.d_model,dropout=dropout)

        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder_norm = LayerNorm(d_model)

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, self.encoder_norm)
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)

        self.fc1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.fc2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, int(self.d_model/4))
        self.fc4 = torch.nn.Linear(int(self.d_model/4), 2)

    
    def src_mask(self, src):

        mask = torch.logical_and(src >= self.NOTE_Velocity_lower_bound, 
                                src <= self.NOTE_Velocity_upper_bound).to(torch.int)

        src = torch.mul(src, mask.logical_not_())

        return src


    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        # src = Pitch_discriminator_infill(src)

        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    

    def discriminate_forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        src = torch.matmul(src, self.token_embedding.weight)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    
    

class Rhythm_Discriminator(Module):

    def __init__(self, vocab, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_distance=64, activation="relu"):
        
        super(Rhythm_Discriminator, self).__init__()

        self.vocab = vocab
        self.ntoken = self.vocab.n_tokens


        self.NOTE_MELODY_lower_bound = self.vocab.token2id['Note-On-MELODY_1']
        self.NOTE_MELODY_upper_bound = self.vocab.token2id['Note-On-MELODY_127']

        self.NOTE_BRIDGE_lower_bound = self.vocab.token2id['Note-On-BRIDGE_1']
        self.NOTE_BRIDGE_upper_bound = self.vocab.token2id['Note-On-BRIDGE_127']

        self.NOTE_PIANO_lower_bound = self.vocab.token2id['Note-On-PIANO_1']
        self.NOTE_PIANO_upper_bound = self.vocab.token2id['Note-On-PIANO_127']

        self.NOTE_Velocity_lower_bound = self.vocab.token2id['Note-Velocity-MELODY_1']
        self.NOTE_Velocity_upper_bound = self.vocab.token2id['Note-Velocity-PIANO_126']


        self.d_model = d_model
        self.nhead = nhead

        self.pos_enc = PositionalEncoding(self.d_model,dropout=dropout)

        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder_norm = LayerNorm(d_model)

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, self.encoder_norm)
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)

        self.fc1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.fc2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, int(self.d_model/4))
        self.fc4 = torch.nn.Linear(int(self.d_model/4), 2)


    def src_mask(self, src):

        mask_part_MELODY = torch.logical_and(src >= self.NOTE_MELODY_lower_bound, 
                                             src <= self.NOTE_MELODY_upper_bound).to(torch.int)
        mask_part_BRIGE = torch.logical_and(src >= self.NOTE_BRIDGE_lower_bound, 
                                             src <= self.NOTE_BRIDGE_upper_bound).to(torch.int)
        mask_part_PIANO = torch.logical_and(src >= self.NOTE_PIANO_lower_bound, 
                                             src <= self.NOTE_PIANO_upper_bound).to(torch.int)

        mask = mask_part_BRIGE + mask_part_MELODY + mask_part_PIANO

        src = torch.mul(src, mask.logical_not_())

        return src



    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    

    
    def discriminate_forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        src = torch.matmul(src, self.token_embedding.weight)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    

    


def calculate_distances_matrix(matrix, target_element, max_distance):

    result_matrix = []

    for row in matrix:
        distances_row = []
        prev_index = -1

        for j, element in enumerate(row):
            if element == target_element:
                prev_index = j
                distances_row.append(0)
            elif prev_index >= 0:
                distances_row.append(j - prev_index)
            else:
                distances_row.append(max_distance)

        result_matrix.append(distances_row)

    return result_matrix

def clip_matrix(matrix, threshold):
    clipped_matrix = matrix.clone()
    clipped_matrix[clipped_matrix > threshold] = threshold
    return clipped_matrix


class Rhythm_Discriminator_with_rpe(Module):

    def __init__(self, vocab, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_distance=63, activation="relu"):
        
        super(Rhythm_Discriminator_with_rpe, self).__init__()

        self.vocab = vocab
        self.ntoken = self.vocab.n_tokens

        self.relative_position_encoding = WordEmbedding(vocab_size=max_distance+1,embed_size=d_model)
        self.max_distance = max_distance


        self.NOTE_MELODY_lower_bound = self.vocab.token2id['Note-On-MELODY_1']
        self.NOTE_MELODY_upper_bound = self.vocab.token2id['Note-On-MELODY_127']

        self.NOTE_BRIDGE_lower_bound = self.vocab.token2id['Note-On-BRIDGE_1']
        self.NOTE_BRIDGE_upper_bound = self.vocab.token2id['Note-On-BRIDGE_127']

        self.NOTE_PIANO_lower_bound = self.vocab.token2id['Note-On-PIANO_1']
        self.NOTE_PIANO_upper_bound = self.vocab.token2id['Note-On-PIANO_127']

        self.NOTE_Velocity_lower_bound = self.vocab.token2id['Note-Velocity-MELODY_1']
        self.NOTE_Velocity_upper_bound = self.vocab.token2id['Note-Velocity-PIANO_126']


        self.d_model = d_model
        self.nhead = nhead

        self.pos_enc = PositionalEncoding(self.d_model,dropout=dropout)

        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder_norm = LayerNorm(d_model)

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, self.encoder_norm)
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)

        self.fc1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.fc2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, int(self.d_model/4))
        self.fc4 = torch.nn.Linear(int(self.d_model/4), 2)


    def src_mask(self, src):

        mask_part_MELODY = torch.logical_and(src >= self.NOTE_MELODY_lower_bound, 
                                             src <= self.NOTE_MELODY_upper_bound).to(torch.int)
        mask_part_BRIGE = torch.logical_and(src >= self.NOTE_BRIDGE_lower_bound, 
                                             src <= self.NOTE_BRIDGE_upper_bound).to(torch.int)
        mask_part_PIANO = torch.logical_and(src >= self.NOTE_PIANO_lower_bound, 
                                             src <= self.NOTE_PIANO_upper_bound).to(torch.int)

        mask = mask_part_BRIGE + mask_part_MELODY + mask_part_PIANO

        src = torch.mul(src, mask.logical_not_())

        return src



    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        rpe_mat = calculate_distances_matrix(src, self.vocab.token2id['Bar'], self.max_distance)
        rpe_mat = torch.tensor(rpe_mat).cuda()
        rpe_mat = clip_matrix(rpe_mat, self.max_distance)
        rpe = self.relative_position_encoding(rpe_mat)

        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)

        src = src+rpe.permute(1, 0, 2)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    

    
    def discriminate_forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        src = torch.matmul(src, self.token_embedding.weight)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info



class Rhythm_Discriminator_with_rpe_val(Module):

    def __init__(self, vocab, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_distance=63, activation="relu"):
        
        super(Rhythm_Discriminator_with_rpe_val, self).__init__()

        self.vocab = vocab
        self.ntoken = self.vocab.n_tokens

        self.relative_position_encoding = WordEmbedding(vocab_size=max_distance+1,embed_size=d_model)
        self.max_distance = max_distance


        self.NOTE_MELODY_lower_bound = self.vocab.token2id['Note-On-MELODY_1']
        self.NOTE_MELODY_upper_bound = self.vocab.token2id['Note-On-MELODY_127']

        self.NOTE_BRIDGE_lower_bound = self.vocab.token2id['Note-On-BRIDGE_1']
        self.NOTE_BRIDGE_upper_bound = self.vocab.token2id['Note-On-BRIDGE_127']

        self.NOTE_PIANO_lower_bound = self.vocab.token2id['Note-On-PIANO_1']
        self.NOTE_PIANO_upper_bound = self.vocab.token2id['Note-On-PIANO_127']

        self.NOTE_Velocity_lower_bound = self.vocab.token2id['Note-Velocity-MELODY_1']
        self.NOTE_Velocity_upper_bound = self.vocab.token2id['Note-Velocity-PIANO_126']


        self.d_model = d_model
        self.nhead = nhead

        self.pos_enc = PositionalEncoding(self.d_model,dropout=dropout)

        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder_norm = LayerNorm(d_model)

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, self.encoder_norm)
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)

        self.fc1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.fc2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, int(self.d_model/4))
        self.fc4 = torch.nn.Linear(int(self.d_model/4), 2)


    def src_mask(self, src):

        mask_part_MELODY = torch.logical_and(src >= self.NOTE_MELODY_lower_bound, 
                                             src <= self.NOTE_MELODY_upper_bound).to(torch.int)
        mask_part_BRIGE = torch.logical_and(src >= self.NOTE_BRIDGE_lower_bound, 
                                             src <= self.NOTE_BRIDGE_upper_bound).to(torch.int)
        mask_part_PIANO = torch.logical_and(src >= self.NOTE_PIANO_lower_bound, 
                                             src <= self.NOTE_PIANO_upper_bound).to(torch.int)

        mask = mask_part_BRIGE + mask_part_MELODY + mask_part_PIANO

        src = torch.mul(src, mask.logical_not_())

        return src



    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        # src = self.src_mask(src)

        rpe_mat = calculate_distances_matrix(src, self.vocab.token2id['Bar'], self.max_distance)
        rpe_mat = torch.tensor(rpe_mat).cuda()
        rpe_mat = clip_matrix(rpe_mat, self.max_distance)
        rpe = self.relative_position_encoding(rpe_mat)

        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)

        src = src+rpe.permute(1, 0, 2)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
    

    
    def discriminate_forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        src = torch.matmul(src, self.token_embedding.weight)
        src = self.pos_enc(src)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info


class Pitch_Discriminator_rpe(Module):

    def __init__(self, vocab, d_model=256, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_distance=63, activation="relu"):
        
        super(Pitch_Discriminator_rpe, self).__init__()

        self.vocab = vocab
        self.ntoken = self.vocab.n_tokens

        self.relative_position_encoding = WordEmbedding(vocab_size=max_distance+1,embed_size=d_model)
        self.max_distance = max_distance


        self.NOTE_MELODY_lower_bound = self.vocab.token2id['Note-On-MELODY_1']
        self.NOTE_MELODY_upper_bound = self.vocab.token2id['Note-On-MELODY_127']

        self.NOTE_BRIDGE_lower_bound = self.vocab.token2id['Note-On-BRIDGE_1']
        self.NOTE_BRIDGE_upper_bound = self.vocab.token2id['Note-On-BRIDGE_127']

        self.NOTE_PIANO_lower_bound = self.vocab.token2id['Note-On-PIANO_1']
        self.NOTE_PIANO_upper_bound = self.vocab.token2id['Note-On-PIANO_127']

        self.NOTE_Velocity_lower_bound = self.vocab.token2id['Note-Velocity-MELODY_1']
        self.NOTE_Velocity_upper_bound = self.vocab.token2id['Note-Velocity-PIANO_126']


        self.d_model = d_model
        self.nhead = nhead

        self.pos_enc = PositionalEncoding(self.d_model,dropout=dropout)

        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder_norm = LayerNorm(d_model)

        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, self.encoder_norm)
        self.token_embedding = WordEmbedding(vocab_size=self.ntoken,embed_size=self.d_model)

        self.fc1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.fc2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.fc3 = torch.nn.Linear(self.d_model, int(self.d_model/4))
        self.fc4 = torch.nn.Linear(int(self.d_model/4), 2)

    
    def src_mask(self, src):

        mask = torch.logical_and(src >= self.NOTE_Velocity_lower_bound, 
                                src <= self.NOTE_Velocity_upper_bound).to(torch.int)

        src = torch.mul(src, mask.logical_not_())

        return src


    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src = self.src_mask(src)

        rpe_mat = calculate_distances_matrix(src, self.vocab.token2id['Bar'], self.max_distance)
        rpe_mat = torch.tensor(rpe_mat).cuda()
        rpe_mat = clip_matrix(rpe_mat, self.max_distance)
        rpe = self.relative_position_encoding(rpe_mat)


        src = src.permute(1, 0)
        src = self.token_embedding(src)
        src = self.pos_enc(src)

        src = src+rpe.permute(1, 0, 2)
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        memory = memory.permute(1, 0, 2)
        info = memory[:,0,:]

        info = F.relu(self.fc1(info))
        info = F.relu(self.fc2(info))
        info = F.relu(self.fc3(info))
        info = self.fc4(info)

        return info
