""" Training Code with fine-graind discriminators

    Author: Zhedong Zhang
    Email: zhedong_zhang@hdu.edu.cn
    Date: 2023/11/03
    
"""
device_str = 'cuda:0'
import shutil
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F

from mymodel import myLM

from preprocess.music_data import getMusicDataset

from preprocess.vocab import Vocab

from parse_arg import *

import time
import os

import logger

from randomness import set_global_random_seed
from torch.utils.tensorboard import SummaryWriter
from discriminator import TransformerDiscriminator, Pitch_Discriminator, Rhythm_Discriminator, Rhythm_Discriminator_with_rpe

# Set the random seed manually for reproducibility.
set_global_random_seed(args.seed)

# create vocab
myvocab = Vocab()

# create directory for training purpose
os.makedirs("./ckpts",exist_ok=True)
os.makedirs("./logs",exist_ok=True)

# create work directory
# while(1):
#     exp_name = input("Enter exp name : ")
#     if os.path.exists(os.path.join("./ckpts", exp_name)):
#         ans = input("work dir exists! overwrite? [Y/N]:")
#         if ans.lower() == "y":
#             break
#     else:
#         break

exp_name = '8_28_debug'
writer = SummaryWriter('./tensorboard_log/tensor_log_{}'.format(exp_name))

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
                'inference.py', 
                'myTransformer.py',
                'randomness.py', 
                'train_multi_discriminator.py',
                'parse_arg.py', 
                'mymodel.py', 
                'preprocess/vocab.py', 
                'preprocess/music_data.py',
                'discriminator.py'
                ]
for x in file_to_save:
    shutil.copyfile(x, os.path.join(checkpoint_folder, "script", x))

# create logger for log
mylogger = logger.logger(filepath=os.path.join(
    checkpoint_folder, "log/log_{}.txt".format(exp_name)),overrite=True)
if os.path.exists("logs/log_{}.txt".format(exp_name)):
    os.remove("logs/log_{}.txt".format(exp_name))
os.link(mylogger.filepath, "logs/log_{}.txt".format(exp_name))
mylogger.log("Exp_dir : {}".format(checkpoint_folder))
mylogger.log("Exp_Name : {}".format(exp_name))


# devices
device = torch.device( device_str if args.cuda else 'cpu')
device_cpu = torch.device('cpu')


# dataset
train_dataset = getMusicDataset(pkl_path="./data_pkl/train_seg2_512.pkl",
                                args=args,
                                vocab=myvocab)


val_dataset = getMusicDataset(pkl_path="./data_pkl/val_seg2_512.pkl",
                                args=args,
                                vocab=myvocab)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=2,
                        shuffle=False,
                        num_workers=4)

# define model
model = myLM(myvocab.n_tokens, d_model=256, num_encoder_layers=6, xorpattern=[0,0,0,1,1,1])

# whether use the pretrain model
if args.pretrain:
    model.load_state_dict(torch.load('trained_model/model_ep2311.pt'))

# # whehter use the discriminator
# if args.discriminator:
#     discriminator = TransformerDiscriminator(myvocab.n_tokens)
#     discriminator.to(device)
#     discriminator.load_state_dict(torch.load('ckpts/Discriminator_test_acc_best.pt',map_location=device_str))
#     # for param in discriminator.parameters():
#     #     param.requires_grad = False

# define and load the pretrain pitch and rhythm discriminator
pitch_discriminator = Pitch_Discriminator(myvocab)
pitch_discriminator.to(device)
pitch_discriminator.load_state_dict(torch.load('trained_model/Melody_Discriminator_test_acc_best.pt',map_location=device_str))

rhythm_discriminator = Rhythm_Discriminator_with_rpe(myvocab)
rhythm_discriminator.to(device)
rhythm_discriminator.load_state_dict(torch.load('trained_model/Rhythm_Discriminator_test_acc_best.pt',map_location=device_str))


mylogger.log("Model hidden dim : {}".format(model.d_model))
mylogger.log("Encoder Layers #{}".format(model.num_encoder_layers))
mylogger.log("Decoder Layers #{}".format(len(model.xorpattern)))
mylogger.log("Decoder Pattern #{}".format(model.xorpattern))
mylogger.log("Batch size #{}".format(args.batch_size))
mylogger.log("lr : {}".format(args.lr))

# optimizer for model
# adam
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# optimizer for discriminator
optimizer_pitch = torch.optim.Adam(pitch_discriminator.parameters(), lr=args.lr/100)
optimizer_rhythm = torch.optim.Adam(rhythm_discriminator.parameters(), lr=args.lr/100)

# scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.max_step,eta_min=args.lr_min)

if not args.restart_point == '':
    # restart from checkpoint
    mylogger.log("Restart from {}".format(args.restart_point))
    model.load_state_dict(torch.load(args.restart_point,map_location=device_str))
    mylogger.log("model loaded")
    optimizer.load_state_dict(torch.load(args.restart_point.replace('model_','optimizer_'),map_location=device_str))
    mylogger.log("optimizer loaded")
    scheduler.load_state_dict(torch.load(args.restart_point.replace('model_','scheduler_'),map_location=device_str))
    mylogger.log("scheduler loaded")

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

# loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

mylogger.log("Using device : {}".format(logger.getCyan(device)))

train_step = 0


def train(epoch_num):
    """train the model

    Args:
        epoch_num (int): epoch number
    """
    global train_step

    model.train()
    start_time = time.time()
    total_loss_regular = 0
    total_loss_pitch = 0
    total_loss_rhythm = 0
    total_loss_pitch_D = 0
    total_loss_rhythm_D = 0
    total_acc = 0
    steps = 0
    for batch_idx, data in enumerate(train_loader):
        print("Epoch {} Step [{}/{}] ".format(epoch_num,
              batch_idx, len(train_loader)), end='')
        
        # optimize the Generator

        data = {key: value.to(device) for key, value in data.items()}

        optimizer.zero_grad()

        data["src_msk"] = data["src_msk"].bool()
        data["tgt_msk"] = data["tgt_msk"].bool()

        tgt_input_msk = data["tgt_msk"][:, :-1]
        tgt_output_msk = data["tgt_msk"][:, 1:]

        data["src"] = data["src"].permute(1, 0)
        data["tgt"] = data["tgt"].permute(1, 0)
        data["tgt_theme_msk"] = data["tgt_theme_msk"].permute(1, 0)

        fullsong_input = data["tgt"][:-1, :]
        fullsong_output = data["tgt"][1:, :]

        att_msk = model.transformer_model.generate_square_subsequent_mask(
            fullsong_input.shape[0]).to(device)

        mem_msk = None


        output = model(
            src=data["src"],
            tgt=fullsong_input,
            tgt_mask=att_msk,
            tgt_label=data["tgt_theme_msk"][:-1, :],
            src_key_padding_mask=data["src_msk"],
            tgt_key_padding_mask=tgt_input_msk,
            memory_mask=mem_msk
        )

        # 2023.6.30 ZZDoog add adversarial loss
        # 将模型的输出进行 Gumbel Softmax 计算成为准离散的One-hot向量
        softmax_output = F.gumbel_softmax(output, hard=False)
        # Softargmax trick,详情见链接https://www.zhihu.com/question/422373907

        # # 计算出本次输出的Discriminator分数
        pitch_discriminator_score = pitch_discriminator.discriminate_forward(softmax_output)
        rhythm_discriminator_score = rhythm_discriminator.discriminate_forward(softmax_output)

        loss_pitch = F.cross_entropy(pitch_discriminator_score, torch.zeros(softmax_output.shape[1]).long().to(device))
        loss_rhythm = F.cross_entropy(rhythm_discriminator_score, torch.zeros(softmax_output.shape[1]).long().to(device))


        loss_regular = criterion(output.view(-1, myvocab.n_tokens),
                         fullsong_output.reshape(-1))
        loss_total = loss_regular + loss_pitch*0.1 #+ loss_rhythm*0.05
        # loss_total = loss_regular + loss_pitch*0.1
        # loss_total = loss_regular + loss_rhythm*0.1
        predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)
        correct = predict.eq(fullsong_output.reshape(-1))
        correct = torch.sum(
            correct * (~tgt_output_msk).reshape(-1).float()).item()
        correct = correct / \
            torch.sum((~tgt_output_msk).reshape(-1).float()).item()
        total_acc += correct
        print("Acc : {:.2f} ".format(correct), end="")


        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # 2023.7.28  optimize the discriminator

        output = model(
            src=data["src"],
            tgt=fullsong_input,
            tgt_mask=att_msk,
            tgt_label=data["tgt_theme_msk"][:-1, :],
            src_key_padding_mask=data["src_msk"],
            tgt_key_padding_mask=tgt_input_msk,
            memory_mask=mem_msk
        )

        # 2023.6.30 ZZDoog add adversarial loss
        # 将模型的输出进行 Gumbel Softmax 计算成为准离散的One-hot向量
        # Softargmax trick,详情见链接https://www.zhihu.com/question/422373907
        softmax_output = F.gumbel_softmax(output, hard=False)
        
        # 计算出本次输出的Discriminator分数
        pitch_discriminator_score = pitch_discriminator.discriminate_forward(softmax_output.detach())
        rhythm_discriminator_score = rhythm_discriminator.discriminate_forward(softmax_output.detach())


        optimizer_pitch.zero_grad()
        optimizer_rhythm.zero_grad()
 
        pitch_discriminator_score_real_sample = pitch_discriminator(fullsong_output.permute(1, 0))
        rhythm_discriminator_score_real_sample = rhythm_discriminator(fullsong_output.permute(1, 0))

        loss_pitch_discriminator = F.cross_entropy(pitch_discriminator_score, 
                                                   torch.ones(softmax_output.shape[1]).long().to(device))+ \
                                   F.cross_entropy(pitch_discriminator_score_real_sample,
                                                   torch.zeros(softmax_output.shape[1]).long().to(device))
        loss_rhythm_discriminator = F.cross_entropy(rhythm_discriminator_score, 
                                                    torch.ones(softmax_output.shape[1]).long().to(device)) + \
                                    F.cross_entropy(rhythm_discriminator_score_real_sample, 
                                                    torch.zeros(softmax_output.shape[1]).long().to(device))
        

        if args.interval_update == True:
        
            # the optimizer is too strong
            if batch_idx % 40 == 0 :

                loss_pitch_discriminator.backward()
                loss_rhythm_discriminator.backward()
                torch.nn.utils.clip_grad_norm_(pitch_discriminator.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm_(rhythm_discriminator.parameters(), args.clip)
                optimizer_pitch.step()
                optimizer_rhythm.step()
        
        else:

            loss_pitch_discriminator.backward()
            loss_rhythm_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(pitch_discriminator.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(rhythm_discriminator.parameters(), args.clip)
            optimizer_pitch.step()
            optimizer_rhythm.step()


        if train_step < args.warmup_step:
            curr_lr = args.lr * train_step / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            scheduler.step()
        

        total_loss_regular += loss_regular.item()
        total_loss_pitch += loss_pitch.item()
        total_loss_rhythm += loss_rhythm.item()
        total_loss_pitch_D += loss_pitch_discriminator.item()
        total_loss_rhythm_D += loss_rhythm_discriminator.item()

        print("Loss : {:.2f} Loss_pitch : {:.2f} Loss_rhythm : {:.2f} lr:{:.4f} Loss_pitch_D : {:.2f} Loss_rhythm_D : {:.2f}".format(
            loss_regular.item(), loss_pitch, loss_rhythm, optimizer.param_groups[0]['lr'], loss_pitch_discriminator.item(), loss_rhythm_discriminator.item()
            ), end='\r')

        steps += 1
        train_step += 1

    train_loss_regular = total_loss_regular/steps
    train_acc = total_acc/steps
    train_loss_pitch = total_loss_pitch/steps
    train_loss_rhythm = total_loss_rhythm/steps
    train_loss_pitch_D = total_loss_pitch_D/steps
    train_loss_rhythm_D = total_loss_rhythm_D/steps

    mylogger.log("Epoch {} lr:{:.4f} train_acc : {:.2f} train_loss : {:.2f} loss_pitch : {:.2f} loss_rhythm : {:.2f} loss_pitch_D : {:.2f} loss_rhythm_D : {:.2f} time:{:.2f} ".format(
                    epoch_num,optimizer.param_groups[0]['lr'], train_acc, train_loss_regular, train_loss_pitch, train_loss_rhythm, 
                    train_loss_pitch_D, train_loss_rhythm_D, time.time()-start_time), end='')
    

    return train_loss_regular, train_loss_pitch, train_loss_rhythm, train_acc, train_loss_pitch_D, train_loss_rhythm_D


def evalulate(epoch_num):
    """evaluate validation set

    Args:
        epoch_num (int): epoch number
    """
    model.eval()
    pitch_discriminator.eval()
    rhythm_discriminator.eval()
    start_time = time.time()
    total_loss = 0
    total_acc = 0
    steps = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # print("Epoch {} Step {}/{} ".format( epoch_num,batch_idx,len(val_loader)),end='')
            optimizer.zero_grad()

            data = {key: value.to(device) for key, value in data.items()}

            data["src_msk"] = data["src_msk"].bool()
            data["tgt_msk"] = data["tgt_msk"].bool()

            tgt_input_msk = data["tgt_msk"][:, :-1]
            tgt_output_msk = data["tgt_msk"][:, 1:]

            data["src"] = data["src"].permute(1, 0)
            data["tgt"] = data["tgt"].permute(1, 0)
            data["tgt_theme_msk"] = data["tgt_theme_msk"].permute(1, 0)

            fullsong_input = data["tgt"][:-1, :]
            fullsong_output = data["tgt"][1:, :]

            att_msk = model.transformer_model.generate_square_subsequent_mask(
                fullsong_input.shape[0]).to(device)

            mem_msk = None

            output = model(
                src=data["src"],
                tgt=fullsong_input,
                tgt_mask=att_msk,
                tgt_label=data["tgt_theme_msk"][:-1, :],
                src_key_padding_mask=data["src_msk"],
                tgt_key_padding_mask=tgt_input_msk,
                memory_mask=mem_msk
            )


            # 2023.6.30 ZZDoog add adversarial loss
            # 将模型的输出进行 Gumbel Softmax 计算成为准离散的One-hot向量
            softmax_output = F.gumbel_softmax(output, hard=False)
            # Softargmax trick,详情见链接https://www.zhihu.com/question/422373907

            # # 计算出本次输出的Discriminator分数
            pitch_discriminator_score = pitch_discriminator.discriminate_forward(softmax_output)
            rhythm_discriminator_score = rhythm_discriminator.discriminate_forward(softmax_output)

            eval_loss_pitch = F.cross_entropy(pitch_discriminator_score, torch.zeros(softmax_output.shape[1]).long().to(device))
            eval_loss_rhythm = F.cross_entropy(rhythm_discriminator_score, torch.zeros(softmax_output.shape[1]).long().to(device))


            pitch_discriminator_score_real_sample = pitch_discriminator(fullsong_output.permute(1, 0))
            rhythm_discriminator_score_real_sample = rhythm_discriminator(fullsong_output.permute(1, 0))

            loss_pitch_discriminator = F.cross_entropy(pitch_discriminator_score, 
                                                    torch.ones(softmax_output.shape[1]).long().to(device))+ \
                                    F.cross_entropy(pitch_discriminator_score_real_sample,
                                                    torch.zeros(softmax_output.shape[1]).long().to(device))
            loss_rhythm_discriminator = F.cross_entropy(rhythm_discriminator_score, 
                                                        torch.ones(softmax_output.shape[1]).long().to(device)) + \
                                        F.cross_entropy(rhythm_discriminator_score_real_sample, 
                                                        torch.zeros(softmax_output.shape[1]).long().to(device))

 

            loss = criterion(output.view(-1, myvocab.n_tokens),
                             fullsong_output.reshape(-1))

            predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)

            correct = predict.eq(fullsong_output.reshape(-1))

            correct = torch.sum(
                correct * (~tgt_output_msk).reshape(-1).float()).item()

            correct = correct / \
                torch.sum((~tgt_output_msk).reshape(-1).float()).item()

            total_acc += correct

            total_loss += loss.item()

            steps += 1

        eval_acc = total_acc/steps
        eval_loss = total_loss/steps

        mylogger.log("val_acc: {:.2f} val_loss : {:.2f}".format(
            eval_acc, eval_loss))
        
        return eval_acc, eval_loss, eval_loss_pitch, eval_loss_rhythm, loss_pitch_discriminator, loss_rhythm_discriminator

start_epoch = 0

if not args.restart_point =='':
    start_epoch = int(args.restart_point.split('_')[-1].split('.')[0][2:]) + 1
    mylogger.log("starting from epoch {}".format(start_epoch))

max_epoch = 5500
mylogger.log("max epoch :{}".format(max_epoch))

best_eval_acc = 0
best_eval_loss = 1e100
best_train_acc = 0
best_train_loss = 1e100
best_train_loss_pitch = 1e100
best_train_loss_rhythm = 1e100

for i in range(start_epoch,max_epoch):

    model.to(device)
    train_loss, train_loss_pitch, train_loss_rhythm, train_acc, train_loss_pitch_D, train_loss_rhythm_D = train(i)
    eval_acc, eval_loss, eval_loss_pitch, eval_loss_rhythm, eval_loss_pitch_discriminator, eval_loss_rhythm_discriminator = evalulate(i)
    model.to(device_cpu)

    writer.add_scalar('loss/train_loss', train_loss, i)
    writer.add_scalar('acc/train_acc', train_acc, i)
    writer.add_scalar('loss/eval_loss', eval_loss, i)
    writer.add_scalar('acc/eval_acc', eval_acc, i)

    writer.add_scalar('loss/train_loss_pitch', train_loss_pitch, i)
    writer.add_scalar('loss/train_loss_rhythm', train_loss_rhythm, i)
    writer.add_scalar('loss/eval_loss_pitch', eval_loss_pitch, i)
    writer.add_scalar('loss/eval_loss_rhythm', eval_loss_rhythm, i)

    writer.add_scalar('loss/train_loss_pitch_Discriminator', train_loss_pitch_D, i)
    writer.add_scalar('loss/train_loss_rhythm_Discriminator', train_loss_rhythm_D, i)
    writer.add_scalar('loss/eval_loss_pitch_Discriminator', eval_loss_pitch_discriminator, i)
    writer.add_scalar('loss/eval_loss_rhythm_Discriminator', eval_loss_rhythm_discriminator, i)

    if i % 50 == 0:

        torch.save(model.state_dict(), os.path.join(
                train_checkpoint_folder, "epoch_{}.pt".format(i)))


    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        if eval_acc >= 0.4 and eval_acc < 0.5:
            torch.save(model.state_dict(), os.path.join(
                eval_checkpoint_folder, "model_acc_40.pt"))
        elif eval_acc >= 0.5 and eval_acc < 0.6:
            torch.save(model.state_dict(), os.path.join(
                eval_checkpoint_folder, "model_acc_50.pt"))    
        elif eval_acc >= 0.6 and eval_acc < 0.7:
            torch.save(model.state_dict(), os.path.join(
                eval_checkpoint_folder, "model_acc_60.pt"))
        elif eval_acc >= 0.7 and eval_acc < 0.8:
            torch.save(model.state_dict(), os.path.join(
                eval_checkpoint_folder, "model_acc_70.pt"))
        elif eval_acc >= 0.8:
            torch.save(model.state_dict(), os.path.join(
                eval_checkpoint_folder, "model_acc_{:.2f}.pt".format(eval_acc)))

            
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        if train_acc >= 0.4 and train_acc < 0.5:
            torch.save(model.state_dict(), os.path.join(
                train_checkpoint_folder, "model_acc_40.pt"))
        elif train_acc >= 0.5 and train_acc < 0.6:
            torch.save(model.state_dict(), os.path.join(
                train_checkpoint_folder, "model_acc_50.pt"))
        elif train_acc >= 0.6 and train_acc < 0.7:
            torch.save(model.state_dict(), os.path.join(
                train_checkpoint_folder, "model_acc_60.pt"))
        elif train_acc >= 0.7 and train_acc < 0.8:
            torch.save(model.state_dict(), os.path.join(
                train_checkpoint_folder, "model_acc_70.pt"))
        elif train_acc >= 0.8 :
            torch.save(model.state_dict(), os.path.join(
                train_checkpoint_folder, "model_acc_{:.2f}.pt".format(train_acc)))


    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        torch.save(model.state_dict(), os.path.join(
            eval_checkpoint_folder, "model_loss_best.pt"))

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), os.path.join(
            train_checkpoint_folder, "model_loss_best.pt"))
        
    if train_loss_pitch < best_train_loss_pitch:
        best_train_loss_pitch = train_loss_pitch
        torch.save(model.state_dict(), os.path.join(
            train_checkpoint_folder, "model_loss_pitch_best.pt"))

    if train_loss_rhythm < best_train_loss_rhythm:
        best_train_loss_rhythm = train_loss_rhythm
        torch.save(model.state_dict(), os.path.join(
            train_checkpoint_folder, "model_loss_rhythm_best.pt"))

