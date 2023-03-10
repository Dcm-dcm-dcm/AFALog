import torch
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np_mask.cuda()
    return np_mask


def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad)
    dim0, dim1, dim2 = src_mask.shape
    temp0 = []
    for i in range(dim0):
        temp1 = []
        for j in range(dim1):
            temp = False
            for k in range(dim2):
                temp |= src_mask[i][j][k]
            temp1.append(temp)
        temp0.append(temp1)
    src_mask = torch.BoolTensor(temp0).unsqueeze(-2).to(opt.device)
    if trg is not None:
        trg.to(opt.device)
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
