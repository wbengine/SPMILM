import os
import sys
import numpy as np
import wb


# exact the vocabulary form the corpus
def GetVocab(fname, v):
    f = open(fname, 'rt')
    for line in f:
        a = line.upper().split()
        a.append('<eos>'.upper())  # sentence tail symbol
        for w in a:
            v.setdefault(w, 0)
            v[w] += 1
    f.close()
    return v


# trans txt corpus to id corpus
def CorpusToID(fread, fwrite, v, unk='<UNK>'):
    print('[w2id] ' + fread + ' -> ' + fwrite)
    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.upper().split()
        a.append('<eos>'.upper())  # sentence tail symbol
        for w in a:
            if w in v:
                f2.write('{} '.format(v[w]))
            else:
                f2.write('{} '.format(v[unk]))
        f2.write('\n')
    f1.close()
    f2.close()


# trans id to txt
def CorpusToW(fread, fwrite, v):
    print('[id2w] ' + fread + ' -> ' + fwrite)
    v1 = [''] * (len(v) + 1)
    for key in v.keys():
        v1[v[key]] = key

    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.split()
        for w in a:
            f2.write('{} '.format(v1[int(w)]))
        f2.write('\n')
    f1.close()
    f2.close()


# write vocabulary
def WriteVocab(fname, v):
    f = open(fname, 'wt')
    for k in sorted(v.keys()):
        f.write('{}\t{}\n'.format(k, v[k]))
    f.close()


# read vocabulary
def ReadVocab(fname):
    v = dict()
    f = open(fname, 'rt')
    for line in f:
        a = line.split()
        v[a[0].upper()] = int(a[1])
    f.close()
    return v


# trans nbest list to id files
def GetNbest(ifile, ofile, v, unk='<UNK>'):
    print('[nbest] ' + ifile + ' -> ' + ofile)
    print('[nbest] v={}'.format(len(v)))
    fin = open(ifile, 'rt')
    fout = open(ofile, 'wt')
    for line in fin:
        a = line.upper().split()[1:]
        a.append('<eos>'.upper())
        for w in a:
            nid = -1
            if w in v:
                nid = v[w]
            elif unk in v:
                nid = v[unk]
            fout.write('{} '.format(nid))
        fout.write('\n')
    fin.close()
    fout.close()


# remove the label of nbest
def rmlabel(ifile, ofile):
    print('[rm-label] {} -> {}'.format(ifile, ofile))
    with open(ifile) as f1, open(ofile, 'wt') as f2:
        for a in [line.split()[1:] for line in f1]:
            f2.write(' '.join(a) + '\n')


# train lstm
# revise the lstm config in file: ./tools/lstm/main.lua
def train(workdir, train, valid, test, config=''):
    write_vocab = workdir + 'vocab'
    write_model = workdir + 'model.lstm'
    write_train = workdir + 'train.id'
    write_valid = workdir + 'valid.id'
    write_test = workdir + 'test.id'

    # get the vocab
    v = dict()
    GetVocab(train, v)
    GetVocab(valid, v)
    # GetVocab(test, v)

    # set the word-id
    n = 0
    for k in sorted(v.keys()):
        v[k] = n + 1
        n += 1

    # write vocab to file
    WriteVocab(write_vocab, v)

    # corpus to id
    CorpusToID(train, write_train, v)
    CorpusToID(valid, write_valid, v)
    CorpusToID(test, write_test, v)

    # train lstm
    cmd = 'th main.lua -vocab {} -train {} -valid {} -test {} -write {} '.format(write_vocab, write_train, write_valid,
                                                                                 write_test, write_model)
    cmd += config
    print(cmd)
    os.system(cmd)
    return write_model


# calculate the PPL of given files
def ppl(workdir, txt, config=''):
    read_vocab = workdir + 'vocab'
    read_model = workdir + 'model.lstm'
    read_txt = txt

    v = ReadVocab(read_vocab)
    write_txt = workdir + read_txt.split('/').pop() + '.pplid'
    CorpusToID(read_txt, write_txt, v)
    cmd = 'th main.lua -vocab {} -read {} -test {} '.format(read_vocab, read_model, write_txt)
    cmd += config
    os.system(cmd)


# rescore the nbest list and get the lmscore, input the txt nbest file
def rescore(workdir, nbest, lmscore, config=''):
    read_vocab = workdir + 'vocab'
    read_model = workdir + 'model.lstm'
    read_nbest_txt = nbest
    write_nbest_id = workdir + read_nbest_txt.split('/').pop() + '.id'
    write_lmscore = lmscore

    v = ReadVocab(read_vocab)
    GetNbest(read_nbest_txt, write_nbest_id, v)
    cmd = 'th main.lua -vocab {} -read {} -nbest {} -score {} '.format(read_vocab, read_model, write_nbest_id,
                                                                       write_lmscore)
    cmd += config
    os.system(cmd)

    return write_lmscore


def addpath(add, path):
    if path[0] == os.sep:
        return path
    if add[-1] != os.sep:
        add += os.sep
    return add + path

class model:
    def __init__(self, bindir, workdir):
        self.bindir = wb.folder(bindir)
        self.workdir = wb.folder(workdir)
        wb.mkdir(workdir)

    def prepare(self, train_txt, valid_txt, test_txt):
        write_vocab = self.workdir + 'vocab'
        write_train = self.workdir + 'train.id'
        write_valid = self.workdir + 'valid.id'
        write_test = self.workdir + 'test.id'

        # get the vocab
        v = dict()
        GetVocab(train_txt, v)
        GetVocab(valid_txt, v)
        GetVocab(test_txt, v)

        # set the word-id
        n = 0
        for k in sorted(v.keys()):
            v[k] = n + 1
            n += 1

        # write vocab to file
        WriteVocab(write_vocab, v)

        # corpus to id
        CorpusToID(train_txt, write_train, v)
        CorpusToID(valid_txt, write_valid, v)
        CorpusToID(test_txt, write_test, v)

    def train(self, write_model, config=''):
        write_vocab = self.workdir + 'vocab'
        write_train = self.workdir + 'train.id'
        write_valid = self.workdir + 'valid.id'
        write_test = self.workdir + 'test.id'

        # train lstm
        save_cwd = os.getcwd()
        cmd = 'th main.lua '
        cmd += ' -vocab ' + addpath(save_cwd, write_vocab)
        cmd += ' -train ' + addpath(save_cwd, write_train)
        cmd += ' -valid ' + addpath(save_cwd, write_valid)
        cmd += ' -test ' + addpath(save_cwd, write_test)
        cmd += ' -write ' + addpath(save_cwd, write_model)
        cmd += ' ' + config
        print(cmd)
        os.chdir(self.bindir)
        os.system(cmd)
        os.chdir(save_cwd)

    def ppl(self, read_model, read_txt, config=''):
        read_vocab = self.workdir + 'vocab'
        v = ReadVocab(read_vocab)

        write_txt = self.workdir + read_txt.split(os.sep).pop() + '.pplid'
        CorpusToID(read_txt, write_txt, v)

        save_cwd = os.getcwd()
        cmd = 'th main.lua '
        cmd += ' -vocab ' + addpath(save_cwd, read_vocab)
        cmd += ' -read ' + addpath(save_cwd, read_model)
        cmd += ' -test ' + addpath(save_cwd, write_txt)
        cmd += ' ' + config
        print(cmd)
        os.chdir(self.bindir)
        s = os.popen(cmd).read()
        os.chdir(save_cwd)

        #get ppl
        idx = s.find('Test set perplexity')
        if idx == -1:
            return 0
        return float(s[idx:].split()[4])

    def rescore(self, read_model, read_nbest_txt, write_lmscore, config=''):
        read_vocab = self.workdir + 'vocab'
        write_nbest_id = self.workdir + read_nbest_txt.split(os.sep).pop() + '.id'

        v = ReadVocab(read_vocab)
        GetNbest(read_nbest_txt, write_nbest_id, v)

        save_cwd = os.getcwd()
        cmd = 'th main.lua '
        cmd += ' -vocab ' + addpath(save_cwd, read_vocab)
        cmd += ' -read ' + addpath(save_cwd, read_model)
        cmd += ' -nbest ' + addpath(save_cwd, write_nbest_id)
        cmd += ' -score ' + addpath(save_cwd, write_lmscore)
        cmd += ' ' + config

        print(cmd)
        os.chdir(self.bindir)
        os.system(cmd)
        os.chdir(save_cwd)