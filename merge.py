import pandas as pd
from random import random
from Levenshtein import distance
from tqdm import tqdm


def generate(name, dir):
    dataset = []
    with open('data/' + dir + '/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n, map(int, ln.strip().split())))
            dataset.append(ln)
    print('Number of seqs({}): {}'.format(name, len(dataset)))
    return dataset


def write_to_samples(line, fp):
    for item in line:
        if item != -1:
            fp.write(str(item) + ' ')
    fp.write('\n')


def calculate_sim(normal_seq, abnormal_seq):
    normal_seq_str = ''
    abnormal_seq_str = ''
    for seq in normal_seq:
        normal_seq_str += str(seq)
    for seq in abnormal_seq:
        abnormal_seq_str += str(seq)
    sim = 1 - (distance(normal_seq_str, abnormal_seq_str) / max(len(normal_seq_str), len(abnormal_seq_str)))
    return sim


def merging(dir, drop_out, dup):
    filename = './data/' + dir + '/selected_samples'
    dis_filename = './data/' + dir + '/enhanced_train'
    dis_file = open(dis_filename, 'w')
    print('merging files, drop_out>=' + str(drop_out))

    struct_data = pd.read_csv(filename, engine='c',
                              na_filter=False, memory_map=True)
    normal_seq_list = []
    abnormal_seq_list = []
    sum = 0
    for idx, row in struct_data.iterrows():
        line = row['seq']
        label = row['label']
        ln = list(map(lambda n: n, map(int, line.strip().split())))
        if label == 0:  # normal
            normal_seq_list.append(ln)
        else:  # abnormal
            abnormal_seq_list.append(ln)

    for i in range(dup):
        for normal_seq in normal_seq_list:  # Normal samples are retained
            write_to_samples(normal_seq, dis_file)
            sum += 1

    # Filter the original training set
    selected_sum = 0
    raw_normal_seq_list = generate('train', dir)
    for raw_normal_seq in tqdm(raw_normal_seq_list, desc="Filtering: "):
        max_sim = 0
        for abnormal_seq in abnormal_seq_list:
            max_sim = max(max_sim, calculate_sim(raw_normal_seq, abnormal_seq))
        if max_sim < drop_out:
            write_to_samples(raw_normal_seq, dis_file)
            sum += 1
            selected_sum += 1
        else:  # random drop out
            ran = random()
            if ran > drop_out:
                write_to_samples(raw_normal_seq, dis_file)
                sum += 1
                selected_sum += 1
    print('selected_seqs_from_raw_dataset: ' + str(selected_sum))
    print('all_seqs: ' + str(sum))
    print('close files')
    dis_file.close()


if __name__ == "__main__":
    merging('HDFS', 0.75)
