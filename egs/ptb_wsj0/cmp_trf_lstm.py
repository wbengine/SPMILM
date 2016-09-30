import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, os.getcwd() + '/../../tools/')
import lstm
import trf
import ngram
import wb

workdir = 'cmp_lms/'
[train, valid, test] = ['./data/ptb/ptb.{}.txt'.format(i) for i in ['train', 'valid', 'test']]
max_repnum = 10
fres = wb.FRes(workdir + 'result.txt')


def get_vocab_list(corpus):
    v = dict()
    for s in corpus:
        with open(s) as f:
            for line in f:
                for w in line.split():
                    v.setdefault(w, 0)
                    v[w] += 1  # count
    return v.keys()


# return selectnum positions from 0 to totalnum-1
def rand_position(totalnum, selectnum):
    pos_list = list(range(totalnum))
    pos_selected = []
    for i in range(selectnum):
        n = random.randint(0, len(pos_list)-1)
        pos_selected.append(pos_list[n])
        del pos_list[n]
    return pos_selected


def prepare():
    wb.mkdir(workdir)
    vlist = get_vocab_list([train, valid, test])
    with open(workdir + "word_list.txt", 'wt') as f:
        for w in vlist:
            f.write(w + '\n')

    fwrite = [open(workdir + 'ptb.test.rep{}.txt'.format(repnum), 'wt') for repnum in range(0,max_repnum)]
    with open(test) as f:
        for line in f:
            a = line.split()
            pos_selected = rand_position(len(a), min(len(a),max_repnum-1))
            for fout in fwrite:
                idx = fwrite.index(fout)
                fout.write(' '.join(a) + '\n')
                if idx < len(pos_selected):
                    a[pos_selected[idx]] = random.choice(vlist)
    for fout in fwrite:
        fout.close()


    # for repnum in range(0,max_repnum):
    #     write_test = workdir + 'ptb.test.rep{}.txt'.format(repnum)
    #     print('output: ' + write_test)
    #
    #     with open(test) as f1, open(write_test, 'wt') as f2:
    #         for line in f1:
    #             a = line.split()
    #             pos_selected = rand_position(len(a), min(len(a),repnum))
    #             for pos in pos_selected:
    #                 a[pos] = random.choice(vlist)
    #             f2.write(' '.join(a) + '\n')

def run_lstm():
    hidden = 250
    dropout = 0
    epoch = 10
    gpu = 1
    runnum = 0
    model = lstm.model('../../tools/lstm', 'lstmlm/')
    read_model = 'lstmlm/h{}_dropout{}_epoch{}.run{}.lstm'.format(hidden, dropout, epoch, runnum)
    write_name = 'LSTM:h{}d{}epoch{}.run{}'.format(hidden, dropout, epoch, runnum)
    config = '-hidden {} -dropout {} -epoch {}  -gpu {}'.format(hidden, dropout, epoch, gpu)
    PPL = [0]*max_repnum

    for repnum in range(0, max_repnum):
        test = workdir + 'ptb.test.rep{}.txt'.format(repnum)
        print('lstm -> ' + test)
        PPL[repnum] = model.ppl(read_model, test, config)
        fres.Add(write_name, ['ppl-rep{}'.format(repnum)], [PPL[repnum]])
    return PPL

def run_trf():
    model = trf.model('../../tools/trf/bin/', 'trflm/')
    class_num = 200
    feat = 'g4_w_c_ws_cs_wsh_csh_tied.fs'
    runnum = 1

    class_vocab = 'trflm/vocab_c{}.list'.format(class_num)
    write_model = 'trflm/trf_c{}_{}.run{}'.format(class_num, feat[0:-3], runnum)
    write_name = '{}'.format(os.path.split(write_model)[1])

    PPL = [0]*max_repnum
    for repnum in range(0, max_repnum):
        test = workdir + 'ptb.test.rep{}.txt'.format(repnum)
        print('trf -> ' + test)
        PPL[repnum] = model.ppl(class_vocab, write_model+'.model', test)
        fres.Add(write_name, ['ppl-rep{}'.format(repnum)], [PPL[repnum]])
    return PPL

def run_ngram():
    order = 5
    model = ngram.model('../../tools/srilm/', 'ngramlm/')
    write_model = 'ngramlm/{}gram.lm'.format(order)
    write_name = 'KN{}'.format(order)

    PPL = [0]*max_repnum
    for repnum in range(0, max_repnum):
        test = workdir + 'ptb.test.rep{}.txt'.format(repnum)
        print('KN5 -> ' + test)
        PPL[repnum] = model.ppl(write_model, order, test)
        fres.Add(write_name, ['ppl-rep{}'.format(repnum)], [PPL[repnum]])
    return PPL

def plot(num):
    color_pack = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure()
    name_pack = fres.Read()
    for name in name_pack:
        v = np.log(fres.Get(name)[1:1+num])
        color = color_pack[name_pack.index(name)]
        plt.plot(list(range(len(v))), v,  color + '-', label=name)

    plt.legend()
    plt.xlabel('replace_num')
    plt.ylabel('log(PPL)')
    plt.show()


if __name__ == '__main__':
    prepare()
    run_trf()
    run_lstm()
    run_ngram()
    plot(10)
