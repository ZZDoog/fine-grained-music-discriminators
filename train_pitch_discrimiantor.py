"""Melody Transformer Training Code

    Author: ZZDoog
    Email: zhedong_zhang@hdu.edu.cn
    Date: 2023.5.22
    
"""

import shutil
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import random_split
import torch.optim

from discriminator import Pitch_Discriminator, FocalLoss, Pitch_Discriminator_rpe

from preprocess.vocab import Vocab
from preprocess.Discriminator_dataset import Discriminator_dataset
from tqdm import tqdm

from parse_arg import *

import time
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

import logger

from randomness import set_global_random_seed
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

exp_name = 'debug'
# writer = SummaryWriter('./tensorboard_log/tensor_log_{}'.format(exp_name))
checkpoint_folder = "./ckpts/{}".format(exp_name)

os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs("./ckpts",exist_ok=True)
os.makedirs("./logs",exist_ok=True)

os.makedirs(os.path.join(
    "./ckpts/", exp_name), exist_ok=True)
os.makedirs(os.path.join(
    "./ckpts/", exp_name, 'train'), exist_ok=True)
os.makedirs(os.path.join(
    "./ckpts/", exp_name, 'eval'), exist_ok=True)
os.makedirs(os.path.join("./ckpts/",
            exp_name, "script"), exist_ok=True)
os.makedirs(os.path.join("./ckpts/",
            exp_name, "script", "preprocess"), exist_ok=True)
os.makedirs(os.path.join("./ckpts/",
            exp_name, "log"), exist_ok=True)

checkpoint_folder = "./ckpts/{}".format(
    exp_name)
train_checkpoint_folder = "./ckpts/{}/train".format(
    exp_name)
eval_checkpoint_folder = "./ckpts/{}/eval".format(
    exp_name)

# copy scripts
file_to_save = [
                'train_pitch_discrimiantor.py',
                'train_rhythm_discrimiantor_val.py',
                'preprocess/Discriminator_dataset.py',
                'discriminator.py'
                ]
for x in file_to_save:
    shutil.copyfile(x, os.path.join(checkpoint_folder, "script", x))



# create vocab
myvocab = Vocab()

BATCH_SIZE = 96
# dataset
Dataset = Discriminator_dataset(real_data_path='data_pkl/discriminator_realdata.pkl',
                                fake_data_path='data_pkl/discriminator_fakedata.pkl')

train_dataset, test_dataset = random_split(dataset=Dataset, lengths=[int(0.7*len(Dataset)), len(Dataset)-int(0.7*len(Dataset))], 
                                           generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=2,
                        shuffle=False,
                        num_workers=4)


# define model
# model = Pitch_Discriminator(myvocab)
model = Pitch_Discriminator(myvocab)
# model = TransformerDiscriminator(myvocab.n_tokens)
model.cuda()

# optimizer
# adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# loss
# criterion = torch.nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')

max_epoch = 10000

best_test_acc = 0
best_train_acc = 0
best_train_loss = 1e100

for train_epoch in range(max_epoch):

    total_loss =0
    num_total = 0
    num_correct = 0
    step = 0

    for data_train, label_train in train_loader:

        model.train()
        data_train = data_train.cuda()
        label_train = label_train.cuda()
        output = model(data_train)

        optimizer.zero_grad()
        loss = criterion(output, label_train)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        num_correct += (output.argmax(1) == label_train).sum().item()
        num_total += len(data_train)


        # print
        sys.stdout.write('Epoch {}/{}, Batch {}/{}, Loss:{:.4f}, Acc:{:.4f}\r'.format(train_epoch, max_epoch, step,
                                                                    len(train_loader), total_loss/step, num_correct/num_total))
        sys.stdout.flush()

    print('Epoch {}/{}: Loss:{:.4f}, Acc:{:.4f}----------------------------\r'.format(
        train_epoch, max_epoch, total_loss/step, num_correct/num_total))
    # writer.add_scalar('train_loss', total_loss/step, train_epoch)
    # writer.add_scalar('acc/train_acc', num_correct/num_total, train_epoch)

    if (total_loss/step) < best_train_loss:
        best_train_loss = total_loss/step
        torch.save(model.state_dict(), os.path.join(
            checkpoint_folder, "Discriminator_loss_best.pt"))

    if (num_correct/num_total) > best_train_acc:
        best_train_acc = num_correct/num_total
        torch.save(model.state_dict(), os.path.join(
            checkpoint_folder, "Discriminator_acc_best.pt"))


    if train_epoch % 5 == 0:

        num_test_total = 0
        num_test_correct = 0

        model.eval()
        with torch.no_grad():
            for data_test, label_test in test_loader:

                data_test = data_test.cuda()
                label_test = label_test.cuda()
                output = model(data_test)

                num_test_correct += (output.argmax(1) == label_test).sum().item()
                num_test_total += len(data_test)

        print('Epoch {}/{}:Test Acc:{:.4f}----------------------------\r'.format(
            train_epoch, max_epoch, num_test_correct/num_test_total))
        # writer.add_scalar('acc/test_acc', num_test_correct/num_test_total, train_epoch)

        if (num_test_correct/num_test_total) > best_test_acc:
            best_test_acc = num_test_correct/num_test_total
            torch.save(model.state_dict(), os.path.join(
                checkpoint_folder, "Discriminator_test_acc_best.pt"))



