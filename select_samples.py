from Models import get_model
import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import os


def write_to_samples_with_label(lines, dir, label):
    filePath = './data/' + dir + '/selected_samples'
    if not os.path.exists(filePath):
        fp = open(filePath, "a")
        fp.write("seq,label\n")
    else:
        fp = open(filePath, "a")
    for line in lines:
        for item in line:
            fp.write(str(item - 2) + ' ')
        fp.write(',' + str(label) + '\n')


def is_fuzzy(predict_seq, output, min_score, opt):
    score = 1.1
    for i in range(len(predict_seq)):
        if output[i + 1] == 1:
            break
        num = (predict_seq[i] == output[i + 1]).nonzero().squeeze(0)[0].tolist()
        score = min(score, (opt.num_class - num) / opt.num_class)
    if min_score - opt.t < score < min_score + opt.t:
        return True
    else:
        return False


def create_masks(src, opt):
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
    return src_mask


def generate(name):
    dataset = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n + 2, map(int, ln.strip().split())))
            dataset.append(ln)
    return dataset


def predict_fuzzy_lines(model, raw_lines, opt, min_score):
    max_len = 0
    window_size = opt.window_size
    inputs = []
    outputs = []
    lines = []
    for line in raw_lines:
        temp_line = []
        for num in line:
            temp_line.append(num)
        lines.append(temp_line)
    for i in range(len(lines)):
        if len(lines[i]) > opt.max_len:
            lines[i] = lines[i][:opt.max_len]
        max_len = max(max_len, len(lines[i]))
    max_len = max(window_size, max_len)
    pad = [1]
    for line in lines:
        line += pad * (max_len - len(line))
    for line in lines:
        input = []
        output = []
        for i in range(len(line) - opt.window_size):
            input.append(line[i:i + window_size])
            output.append(line[i + window_size])
        inputs.append(input)
        outputs.append([2] + output)

    src = torch.FloatTensor(inputs).cuda()
    src_mask = create_masks(src, opt)
    e_outputs = model.encoder(src, src_mask)
    predict_seqs = []
    for i in range(len(outputs)):
        predict_seqs.append([])
    outputs_tensor = torch.LongTensor(outputs).cuda()
    for i in range(1, len(outputs[0])):
        trg_mask = nopeak_mask(i, opt)
        outputs_in = outputs_tensor[:, :i]
        out = model.out(model.decoder(outputs_in,
                                      e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        for j in range(len(out)):
            score, seq = out[j][i - 1].data.topk(opt.num_class)
            predict_seqs[j].append(seq)
    fuzzy_lines = []
    for i in range(len(predict_seqs)):
        if is_fuzzy(predict_seqs[i], outputs[i], min_score, opt):
            fuzzy_lines.append(raw_lines[i])
    return fuzzy_lines


def select_samples(unlabeled_samples_normal, unlabeled_samples_abnormol, model_file, opt):
    unlabeled_samples_normal_loader = generate(unlabeled_samples_normal)
    unlabeled_samples_abnormal_loader = generate(unlabeled_samples_abnormol)
    model = get_model(opt, opt.window_size, opt.num_class, False, model_file=model_file)
    model.eval()
    normal_seqs = 0
    abnormal_seqs = 0
    min_score = opt.min_score
    batch = opt.batch

    # Pool-based sample selection
    with torch.no_grad():
        for i in range(0, len(unlabeled_samples_normal_loader), batch):
            right = min(len(unlabeled_samples_normal_loader), i + batch)
            fuzzy_lines = predict_fuzzy_lines(model, unlabeled_samples_normal_loader[i:right], opt, min_score)
            normal_seqs += len(fuzzy_lines)
            write_to_samples_with_label(fuzzy_lines, opt.dataset, 0)

    with torch.no_grad():
        for i in range(0, len(unlabeled_samples_abnormal_loader), batch):
            right = min(len(unlabeled_samples_abnormal_loader), i + batch)
            fuzzy_lines = predict_fuzzy_lines(model, unlabeled_samples_abnormal_loader[i:right], opt, min_score)
            abnormal_seqs += len(fuzzy_lines)
            write_to_samples_with_label(fuzzy_lines, opt.dataset, 1)
    print('selected_normal_seqs: ' + str(normal_seqs))
    print('selected_abnormal_seqs: ' + str(abnormal_seqs))


def select_samples_run(opt):
    # remove file
    if os.path.exists('./data/' + opt.dataset + '/selected_samples'):
        os.remove('./data/' + opt.dataset + '/selected_samples')
    unlabeled_samples_normal = './data/' + opt.dataset + '/unlabeled_samples_normal'
    unlabeled_samples_abnormal = './data/' + opt.dataset + '/unlabeled_samples_abnormal'
    model_file = './models/' + opt.dataset + '/pre_train'
    select_samples(unlabeled_samples_normal, unlabeled_samples_abnormal, model_file, opt)
