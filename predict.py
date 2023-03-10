from Models import get_model
from Batch import nopeak_mask
import torch.nn.functional as F
import torch
import argparse
from tqdm import tqdm

def generate(name, window_size):
    dataset = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n + 2, map(int, ln.strip().split())))
            ln = ln + [1] * (window_size + 1 - len(ln))
            dataset.append(ln)
    return dataset

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

# save FP
def save_fp(fp_seq, dataset):
    save_url = './data/' + dataset + '/false/' + 'FP'
    save_file = open(save_url, 'w')
    for lines in fp_seq:
        for item in lines:
            save_file.write(str(item - 2))
            save_file.write(' ')
        save_file.write('\n')
    save_file.close()

# save FN
def save_fn(tp_seq, test_abnormal, dataset):
    save_url = './data/' + dataset + '/false/' + 'FN'
    save_file = open(save_url, 'w')
    data = []
    fn_seq = []
    tp_seq_set = set([tuple(t) for t in tp_seq])
    with open(test_abnormal, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n + 2, map(int, ln.strip().split())))
            data.append(tuple(ln))
    for lines in data:
        if lines not in tp_seq_set:
            fn_seq.append(lines)
    for lines in fn_seq:
        for item in lines:
            save_file.write(str(item - 2))
            save_file.write(' ')
        save_file.write('\n')
    save_file.close()

def anomaly_seqs_num(predict_seq, output, min_score, opt):
    abnormal_seq = 0
    for i in range(len(predict_seq)):
        if output[i + 1] == 1:
            break
        num = (predict_seq[i] == output[i + 1]).nonzero().squeeze(0)[0].tolist()
        score = (opt.num_class - num) / opt.num_class
        if score < min_score:
            abnormal_seq += 1
    return abnormal_seq


def predict_anomaly_num(model, lines, opt, min_score):
    max_len = 0
    anomaly_num = 0
    window_size = opt.window_size
    inputs = []
    outputs = []
    copy_lines = []
    for line in lines:
        copy_temp = []
        for i in line:
            copy_temp.append(i)
        copy_lines.append(copy_temp)
    for i in range(len(lines)):
        if len(lines[i]) > opt.max_len:
            lines[i] = lines[i][:opt.max_len]
        max_len = max(max_len, len(lines[i]))
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
    anomaly_seqs = []
    for i in range(len(predict_seqs)):
        if anomaly_seqs_num(predict_seqs[i], outputs[i], min_score, opt) >= opt.threshold:
            anomaly_num += 1
            anomaly_seqs.append(copy_lines[i])
    return anomaly_num, anomaly_seqs

def predict(test_normol, test_abnormol, model_file, opt):
    window_size = opt.window_size
    test_normal_loader = generate(test_normol, window_size)
    test_abnormal_loader = generate(test_abnormol, window_size)
    model = get_model(opt, opt.window_size, opt.num_class, False, model_file=model_file)
    model.eval()
    fp_seq = []
    tp_seq = []
    TP = 0
    FP = 0
    min_score = opt.min_score
    batch = opt.batch
    # Test the model
    with torch.no_grad():
        for i in tqdm(range(0, len(test_normal_loader), batch), desc='testing'):
            right = min(len(test_normal_loader), i + batch)
            anomaly_num, anomaly_seqs = predict_anomaly_num(model, test_normal_loader[i:right], opt, min_score)
            FP += anomaly_num
            fp_seq += anomaly_seqs
    with torch.no_grad():
        for i in tqdm(range(0, len(test_abnormal_loader), batch), desc='testing'):
            right = min(len(test_abnormal_loader), i + batch)
            anomaly_num, anomaly_seqs = predict_anomaly_num(model, test_abnormal_loader[i:right], opt, min_score)
            TP += anomaly_num
            tp_seq += anomaly_seqs
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / max((TP + FP), 1)
    R = 100 * TP / max((TP + FN), 1)
    F1 = 2 * P * R / max((P + R), 1)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, P, R, F1))
    print('Finished Predicting')


def test(opt):
    predict('./data/' + opt.dataset + '/test_normal', './data/' + opt.dataset + '/test_abnormal',
            'models/' + opt.dataset + '/model_weights', opt)

def test_by_model_name(opt, model_name):
    predict('./data/' + opt.dataset + '/test_normal', './data/' + opt.dataset + '/test_abnormal',
            'models/' + opt.dataset + '/'+model_name+'.pt', opt)


def test_pre_train(opt):
    predict('./data/' + opt.dataset + '/test_normal', './data/' + opt.dataset + '/test_abnormal',
            'models/' + opt.dataset + '/pre_train', opt)


def test_re_train(opt):
    predict('./data/' + opt.dataset + '/test_normal', './data/' + opt.dataset + '/test_abnormal',
            'models/' + opt.dataset + '/re_train', opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=150)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-src_pad', type=int, default=1)
    parser.add_argument('-trg_pad', type=int, default=1)
    parser.add_argument('-window_size', type=int, default=4)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-num_class', type=int, default=33)
    parser.add_argument('-threshold', type=int, default=1)
    parser.add_argument('-load_weights', default='./models')

    opt = parser.parse_args()
    opt.src_pad = torch.LongTensor([1] * opt.window_size).cuda()
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
