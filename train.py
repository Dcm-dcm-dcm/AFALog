import argparse
import torch
from Models import get_model
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


def get_len(dataset):
    for i, b in enumerate(dataset):
        pass
    return i


# generate dataset
def generate(name, window_size, max_len):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            temp_input = [1] * window_size
            temp_output = [1]
            input = []
            output = [[2]]
            num_sessions += 1
            line = list(map(lambda n: n + 2, map(int, line.strip().split())))
            if len(line) > max_len:
                line = line[:max_len]
            line = line + [1] * (window_size + 1 - len(line))

            for i in range(len(line) - window_size):
                input.append(line[i:i + window_size])
                output.append([line[i + window_size]])
            inputs.append(input)
            outputs.append(output)
    for i in inputs:
        i += [temp_input] * (max_len - len(i))
    for i in outputs:
        i += ([temp_output] * (max_len - len(i) - 1) + [[3]])
    tens_a = torch.LongTensor(inputs)
    tens_b = torch.LongTensor(outputs)
    dataset = TensorDataset(tens_a, tens_b)
    return dataset, max_len

# generate dataset
def generate_by_list(lines, window_size, max_len):
    num_sessions = 0
    inputs = []
    outputs = []
    for line in lines:
        temp_input = [1] * window_size
        temp_output = [1]
        input = []
        output = [[2]]
        num_sessions += 1
        line = list(line)
        if len(line) > max_len:
            line = line[:max_len]
        line = line + [1] * (window_size + 1 - len(line))

        for i in range(len(line) - window_size):
            input.append(line[i:i + window_size])
            output.append([line[i + window_size]])
        inputs.append(input)
        outputs.append(output)

    for i in inputs:
        i += [temp_input] * (max_len - len(i))
    for i in outputs:
        i += ([temp_output] * (max_len - len(i) - 1) + [[3]])
    tens_a = torch.LongTensor(inputs)
    tens_b = torch.LongTensor(outputs)
    dataset = TensorDataset(tens_a, tens_b)
    return dataset, max_len


def calculate_num(line, num_class):
    result = 0
    for i in line:
        result *= num_class
        result += i
    return result


def train_model(model, opt, dataloader, train_len, save_file):
    print("training model...")
    model.train()
    for epoch in range(opt.epochs):
        total_loss = 0
        for i, (src, trg) in tqdm(enumerate(dataloader), total=len(dataloader), desc="epoch " + str(epoch + 1) + ": "):
            trg = trg.squeeze(2)
            if opt.device == 0:
                src, trg = src.cuda(), trg.cuda()
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / train_len
        print("epoch " + str(epoch + 1) + " complete, loss = " + str(avg_loss))
    torch.save(model.state_dict(), save_file)


def train(opt):
    model = get_model(opt, opt.window_size, opt.num_class, True)

    seq_dataset, max_len = generate('./data/' + opt.dataset + '/train', opt.window_size, opt.max_len)
    dataloader = DataLoader(seq_dataset, batch_size=opt.batchsize, shuffle=True, pin_memory=True)
    train_len = get_len(dataloader)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    train_model(model, opt, dataloader, train_len, 'models/' + opt.dataset + '/model_weights')


def train_by_input(opt, model_name, seq_list):
    model = get_model(opt, opt.window_size, opt.num_class, True)

    seq_dataset, max_len = generate_by_list(seq_list, opt.window_size, opt.max_len)
    dataloader = DataLoader(seq_dataset, batch_size=opt.batchsize, shuffle=True, pin_memory=True)
    train_len = get_len(dataloader)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    train_model(model, opt, dataloader, train_len, 'models/' + opt.dataset + '/'+model_name+'.pt')


def pre_train(opt):
    model = get_model(opt, opt.window_size, opt.num_class, True)

    seq_dataset, max_len = generate('./data/' + opt.dataset + '/train', opt.window_size, opt.max_len)
    dataloader = DataLoader(seq_dataset, batch_size=opt.batchsize, shuffle=True, pin_memory=True)
    train_len = get_len(dataloader)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    train_model(model, opt, dataloader, train_len, 'models/' + opt.dataset + '/pre_train')


def re_train(opt):
    model_file = 'models/' + opt.dataset + '/pre_train'
    model = get_model(opt, opt.window_size, opt.num_class, False, model_file=model_file)

    seq_dataset, max_len = generate('./data/' + opt.dataset + '/enhanced_train', opt.window_size, opt.max_len)
    dataloader = DataLoader(seq_dataset, batch_size=opt.batchsize, shuffle=True, pin_memory=True)
    train_len = get_len(dataloader)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    train_model(model, opt, dataloader, train_len, 'models/' + opt.dataset + '/re_train')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=8)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=150)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', type=bool, default=True)
    parser.add_argument('-src_pad', type=int, default=1)
    parser.add_argument('-trg_pad', type=int, default=1)
    parser.add_argument('-window_size', type=int, default=4)
    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-num_class', type=int, default=33)
    parser.add_argument('-dataset', default="HDFS")

    opt = parser.parse_args()
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.src_pad = [1] * opt.window_size
    opt.src_pad = torch.tensor(opt.trg_pad).cuda()
    train(opt)


if __name__ == "__main__":
    main()
