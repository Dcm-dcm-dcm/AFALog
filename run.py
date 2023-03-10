from predict import *
from train import *
from select_samples import *
from merge import *
from adjustment import adjustment
import argparse
import torch

import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default="BGL")
    parser.add_argument('-epochs', type=int, default=8)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-drop_out', type=int, default=0.8)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-batch', type=int, default=512)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-src_pad', type=int, default=1)
    parser.add_argument('-trg_pad', type=int, default=1)
    parser.add_argument('-window_size', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=200)
    parser.add_argument('-num_class', type=int, default=435)
    parser.add_argument('-threshold', type=int, default=1)
    parser.add_argument('-min_score', type=int, default=0.8)
    parser.add_argument('-t', type=int, default=0.02)
    parser.add_argument('-rands', type=int, default=10)
    parser.add_argument('-dup_times', type=int, default=10)
    parser.add_argument('-low', type=float, default=0.7)
    parser.add_argument('-up', type=float, default=0.9)

    opt = parser.parse_args()
    opt.device = 0 if torch.cuda.is_available() else 1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.src_pad = torch.LongTensor([1] * opt.window_size).cuda()
    train_start_time = time.time()
    pre_train(opt)
    select_samples_run(opt)
    merging(opt.dataset, opt.drop_out, opt.dup_times)
    re_train(opt)
    opt.min_score = adjustment(opt, 're_train', opt.low, opt.up)
    test_re_train(opt)

