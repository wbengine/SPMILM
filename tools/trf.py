import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import wb


def System(cmd):
    tool = cmd.split()[0]
    if not os.path.exists(tool) and not os.path.exists(tool + '.exe'):
        print('Cannot find {}, Make EXE'.format(tool))
        curdir = os.getcwd()
        [toolsdir, toolsexe] = os.path.split(tool)
        toolsdir += '/..'
        os.chdir(toolsdir)
        os.system('make all')
        os.chdir(curdir)
    os.system(cmd)


def GetVocab(ftxt, v):
    f = open(ftxt)
    for line in f:
        a = line.upper().split()
        for w in a:
            v.setdefault(w, 0)
            v[w] += 1
    f.close()

    # sort vocab
    vid = 0
    for w in sorted(v.keys()):
        v[w] = vid
        vid += 1


def ReadVocab(fvocab):
    v = dict()
    f = open(fvocab)
    for line in f:
        a = line.split()
        v[a[1]] = int(a[0])
    f.close()
    return v


def WriteVocab(fvocab, v):
    vlist = sorted(v.items(), key=lambda d: d[1])
    f = open(fvocab, 'wt')
    for w, vid in vlist:
        f.write('{}\t{}\n'.format(vid, w))
    f.close()


def CorpusToID(ftxt, fid, v, unk='<UNK>'):
    with open(ftxt, 'rt') as f1, open(fid, 'wt') as f2:
        for line in f1:
            aid = []
            for w in line.upper().split():
                if w in v:
                    aid.append(str(v[w]))
                elif unk in v:
                    aid.append(str(v[unk]))
                else:
                    print('[NbestToID]: {} is not in vocab'.format(w))
            f2.write(' '.join(aid) + '\n')


def NbestToID(fnbest, fout, v, unk='<UNK>'):
    with open(fnbest, 'rt') as f1, open(fout, 'wt') as f2:
        for line in f1:
            aid = []
            for w in line.upper().split()[1:]:
                if w in v:
                    aid.append(str(v[w]))
                elif unk in v:
                    aid.append(str(v[unk]))
                else:
                    print('[NbestToID]: {} is not in vocab'.format(w))
            f2.write(' '.join(aid) + '\n')

#count the max-len of a txt file
def FileMaxLen(fname):
    maxlen = 0
    with open(fname, 'rt') as f:
        for line in f:
            maxlen = max(maxlen, len(line.split()))
    return maxlen


# read the ExValues in log files
def ReadLog(fname):
    t = 0
    n = 0
    value = []
    m = []
    f = open(fname)
    for line in f:
        if 'ExValues' in line:
            beg = line.find('{') + 1
            end = line.find('}', beg)
            a = line[beg:end].split()
            if len(a) > 0:
                t += 1
                value.append(a)
                for i in range(len(a)):
                    if i >= len(m):
                        m.append([])
                    m[i].append(a[i])
    f.close()
    return m


# plot the loglikelihood from log files
# baseline is [ ['KN5', 1,2,3,4,5,6 ], ['RNN',1,2,3,4,5,6] ]
def PlotLog(name_pack, baseline=[]):
    color_pack = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    tmax = 0
    plt.figure()
    # plot
    for i in range(len(name_pack)):
        [path, name] = os.path.split(name_pack[i])
        color = color_pack[i % len(color_pack)]
        print(path, name)
        v = ReadLog(name_pack[i] + '.log')
        if len(v) <= 0:
            continue
        tmax = max(tmax, len(v[0]))
        plt.plot(list(range(1, len(v[0]) + 1)), v[0], color + '-', label=name + ' [train]')
        plt.plot(list(range(1, len(v[1]) + 1)), v[1], color + '--', label=name + ' [valid]')

    # plot baseline
    if len(baseline) > 0:
        for b in baseline:
            print('baseline = ' + b[0])
            plt.plot([b[1]] * tmax, 'r-.', label=b[0] + ' [train]')
            plt.plot([b[2]] * tmax, 'g-.', label=b[0] + ' [valid]')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('-LL')
    plt.show()


class model:
    def __init__(self, bindir, workdir):
        self.bindir = wb.folder(bindir)
        self.workdir = wb.folder(workdir)
        wb.mkdir(workdir)

    def WordCluster(self, fvocab, fid, class_num, write_list):
        bExist = False
        if os.path.exists(write_list):
            # compare the exist vocab and the generated vocab
            print('new vocab   = {}'.format(fvocab))
            print('exist vcoab = {}'.format(write_list))
            bExist = True
            with open(fvocab) as f1, open(write_list) as f2:
                for linea,lineb in zip(f1, f2):
                    a = linea.split()
                    b = lineb.split()
                    if a[0] != b[0]:
                        print('The exist vocab_list is not match to current task!!')
                        print(linea)
                        print(lineb)
                        print(a[0], b[0])
                        print(a[1], b[1])
                        bExist = False
                        break
        if bExist:
            print('The exist class-vocab match to current task, skip cluster process!')
            return

        # cluster
        cmd = self.bindir + 'word-cluster -txt {} -num {} -tag-vocab {} '.format(fid, class_num, fid + '.vocab')
        System(cmd)

        print('output vocabulary to ' + write_list)
        with open(fvocab) as fv, open(fid + '.vocab') as ftag, open(write_list, 'wt') as fw:
            for linea, lineb in zip(fv, ftag):
                a = linea.split()
                b = lineb.split()
                fw.write('{}\t{}\tclass={}\n'.format(a[0], a[1], b[2]))

    def prepare(self, train, valid, test, class_num):
        write_vocab = self.workdir + 'vocab.list'
        write_vocab_c = self.workdir + 'vocab_c{}.list'.format(class_num)
        write_train = self.workdir + 'train.id'
        write_valid = self.workdir + 'valid.id'
        write_test = self.workdir + 'test.id'

        # get vocabulary
        v = dict()
        GetVocab(train, v)
        GetVocab(valid, v)
        # GetVocab(test, v)
        WriteVocab(write_vocab, v)
        CorpusToID(train, write_train, v)
        CorpusToID(valid, write_valid, v)
        CorpusToID(test, write_test, v)

        # get class
        self.WordCluster(write_vocab, write_train, class_num, write_vocab_c)

    def train(self, config):
        cmd = self.bindir + 'trf-satrain ' + config
        System(cmd)

    def train_ml(self, config):
        cmd = self.bindir + 'trf-mltrain ' + config
        System(cmd)

    # return the -LL, if exists
    def use(self, config, bPrint=True):
        write_log = self.workdir + 'trf_model_use.log'
        cmd = self.bindir + 'trf ' + config + ' -log {}'.format(write_log)
        if bPrint:
            System(cmd)
        else:
            os.popen(cmd).read()  # if not read(), the will return before the process finished (window) !!

        with open(write_log) as f:
            s = f.read()
            idx = s.find('-LL = ')
            if idx != -1:
                return float(s[idx:].split()[2])
        return 0

    # calculate the ppl of a txt file
    def ppl(self, vocab, model, txt, isnbest=False):
        list_vocab = self.workdir + 'vocab.list'
        write_id = self.workdir + os.path.split(txt)[-1] + '.pplid'
        if isnbest:
            NbestToID(txt, write_id, ReadVocab(list_vocab))
        else:
            CorpusToID(txt, write_id, ReadVocab(list_vocab))

        cmd = self.bindir + 'trf '
        cmd += ' -vocab {} -read {} -test {}'.format(vocab, model, write_id)
        s = os.popen(cmd).read()
        idx = s.find('-LL = ')
        if idx != -1:
            LL = float(s[idx:].split()[2])
            return wb.LL2PPL(-LL, write_id)
        else:
            print('[ERROR]!!')
            print(s)


    def get_last_value(self, log_file):
        with open(log_file) as f:
            v = []
            for line in f:
                if line.find('ExValues') != -1:
                    v = [float(i) for i in line[line.find('ExValues'):].split()[1:4]]
        return v
