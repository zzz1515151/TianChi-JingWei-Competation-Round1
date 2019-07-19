from __future__ import print_function
import sys
import argparse
import os
import torch 
import time
import imp
import numpy as np
import datetime
from torch import nn, optim
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.AverageMeter import AverageMeter
from utils.logger import logger 
import utils.utils as utils
from model_1 import *
from dataloader.dataset import *

parser = argparse.ArgumentParser(description='jingwei')
parser.add_argument("--exp", type = str, default = "", help = "experiment")
parser.add_argument("--num_workers", type = int, default = 16, help = "num_workers")
parser.add_argument("--checkpoint", type = int, default = 0, help = "load checkpoint")
parser.add_argument('--gpu', type = str, default = "0", help = 'choose GPU')
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

exp_config = os.path.join(".", "config", args.exp + ".py")
exp_dir = os.path.join("../data/jingwei", args.exp)
exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)
#读取参数
config = imp.load_source("", exp_config).config
#tensorboard && logger
now_str = datetime.datetime.now().__str__().replace(' ','_')

logger_path = os.path.join(exp_log_dir, now_str + ".log")
logger = logger(logger_path).get_logger()

os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'

train_config = config['train_config']

logger.info('preparing data......')

train_dataset = jingwei_train_dataset(
                csv_root = train_config['csv_root'],
                        )
trainloader = DataLoader(
                dataset = train_dataset,
                batch_size = train_config['batch_size'],
                shuffle = True,
                num_workers = args.num_workers,
                drop_last = True
                )
logger.info('data done!')

net_opt = config['net']
net = DeepLabV3_4(net_opt)
net = net.cuda()
net = nn.DataParallel(net, device_ids=[0, 1, 2])

optim_opt = config["optim"]

optimizer = optim.SGD(
                net.parameters(),                                                    
                lr = optim_opt["lr"], \
                momentum = optim_opt["momentum"], \
                nesterov = optim_opt["nesterov"], \
                weight_decay=optim_opt['weight_decay'])

if args.checkpoint > 0 :
    checkpoint_name = args.exp + "_epoch" + args.checkpoint
    checkpoint_path = os.path.join(exp_dir, net_checkpoint_name)
    assert(os.path.exists(net_checkpoint_path))
    try:        
        checkpoint = torch.load(net_checkpoint_path)
        network.load_state_dict(checkpoint["network"])        
        logger.info("Load net checkpoint epoch {}".format(args.checkpoint))
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Load optimizer checkpoint epoch {}".format(args.checkpoint))
    except:
        logger.info("Can not load checkpoint from {}".format(net_checkpoint_path))

weight = torch.Tensor([0.2, 1.0, 1.0, 1.0]).cuda()

if args.checkpoint > 0:
    iter = args.checkpoint * len(trainloader) 
else:
    iter = 0
    
train_loss_1 = AverageMeter()
train_acc = AverageMeter()
train_IOU = AverageMeter()
train_back_IOU = AverageMeter()
val_loss_1 = AverageMeter()
val_acc = AverageMeter()
val_IOU = AverageMeter()
val_back_IOU = AverageMeter()

def train(epoch):
    global iter   
    max_iter = config['num_epochs'] * len(trainloader) 
    train_loss_1.reset()         
    train_acc.reset()    
    train_IOU.reset()
    train_back_IOU.reset()
    net.train()
    for idx, batch in enumerate(trainloader):
        end = time.time()
        new_lr = utils.polynomial_decay(optim_opt['lr'], iter, max_iter, power = 0.9, end_learning_rate = 1e-4) 
        utils.adjust_learning_rate(optimizer, new_lr)
        image = batch[0].cuda()
        instance_label = batch[1].cuda()           
        optimizer.zero_grad()
        prob_output = net(image)        

        loss1 = F.cross_entropy(prob_output, instance_label, weight = weight)  

        total_loss = loss1
        total_loss.backward()
        optimizer.step()
        ####################################
        train_loss_1.update(loss1.item())         

        acc, IOU, back_IOU = utils.compute_accuracy(prob_output, instance_label)             
        train_acc.update(acc)        
        train_IOU.update(IOU)
        train_back_IOU.update(back_IOU)

        if idx % config['display_step'] == 0:
            logger.info('==> Iteration [{}][{}/{}][{}/{}]: loss1: {:.4f} ({:.4f})  lr:{:.4f} acc: {:.4f} ({:.4f}) IOU: {:.4f} ({:.4f}) back_IOU: {:.4f} ({:.4f}) time: {:.4f}'.format(
                                                                            epoch + 1,
                                                                            idx,
                                                                            len(trainloader),
                                                                            iter,
                                                                            max_iter,
                                                                            loss1.item(),
                                                                            train_loss_1.avg,                                                                                             
                                                                            new_lr,                                                                      
                                                                            acc,
                                                                            train_acc.avg,     
                                                                            IOU,
                                                                            train_IOU.avg, 
                                                                            back_IOU,
                                                                            train_back_IOU.avg, 
                                                                            time.time() - end,                            
                                                                            ))
        

        iter += 1

logger.info("training Status: ") 
logger.info(config) 
assert(args.checkpoint < config['num_epochs'])
for epoch in range(args.checkpoint, config['num_epochs']):
    logger.info("Experiment:{}".format(args.exp))
    logger.info("Begin training epoch {}".format(epoch + 1))
    train(epoch) 
    checkpoint_name = args.exp + '_epoch' + str(epoch + 1)
    checkpoint_path = os.path.join(exp_dir, checkpoint_name)
    ckpt = {'epoch': epoch + 1, 'network': net.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(ckpt, checkpoint_path)  





