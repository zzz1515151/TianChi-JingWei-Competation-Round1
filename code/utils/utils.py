import torch
import numpy as np 
import collections
"""
Some Scripts 
@author: Zeyu Song
"""

def polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate = 1e-4, power = 1.0):
    global_step = min(global_step, decay_steps)
    decayed_lr = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ** power + end_learning_rate
    return decayed_lr

def compute_accuracy(output, target):
    batch_size = output.shape[0]
    class_num = output.shape[1]
    h = output.shape[2]
    w = output.shape[3]
    avg_acc, avg_IOU, avg_back_IOU = 0.0, 0.0, 0.0
    for i in range(batch_size):
        _, max_map = torch.max(output[i, :, :, :], dim = 0)
        target_i = target[i, :, :]        
        acc = 0.0       
        num_classes = 0
        IOU = 0.0 
        back_IOU = 0.0       
        num_union = 0
        for c in range(class_num):
            num_c = float(torch.sum(torch.eq(target_i, c)))
            num_c_pred = float(torch.sum(torch.eq(max_map, c)))
            num_true = float(torch.sum(torch.eq(max_map, c) * torch.eq(target_i, c)))
            union_size = num_c + num_c_pred - num_true
            if num_c > 0 or num_c_pred > 0 and c != 0 :
                IOU = IOU + num_true / union_size
                num_union = num_union + 1
            if num_c > 0 or num_c_pred > 0 and c == 0 :
                back_IOU = num_true / union_size
            if num_c > 0 :
                acc = acc + num_true / num_c   
                num_classes += 1   
        if num_classes != 0:
            acc = acc / num_classes
        avg_acc = avg_acc + acc
        if num_union != 0:
            IOU = IOU / num_union
        avg_IOU = avg_IOU + IOU 
        avg_back_IOU = avg_back_IOU + back_IOU       
    avg_acc = avg_acc / batch_size    
    avg_IOU = avg_IOU / batch_size
    avg_back_IOU = avg_back_IOU / batch_size
    return avg_acc, avg_IOU, avg_back_IOU

def parallel_transfer(ckpt):
    new_ckpt = {}
    for item, value in ckpt.items():
        name = '.'.join(item.split('.')[1:])
        new_ckpt[name] = value
    return new_ckpt


def adjust_learning_rate(optimizer, lr):
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]['lr'] = lr
    else:
        # enlarge the lr at the head
        optimizer.param_groups[0]['lr'] = lr
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * 10