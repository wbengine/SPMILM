import os
import sys
import math
import numpy as np
import wb


# exact the vocabulary form the corpus
def GetVocab(fname, v):
    f = open(fname, 'rt')
    for line in f:
        a = line.upper().split()
        for w in a:
            v.setdefault(w, 0)
            v[w] += 1
    f.close()
    return v


# set the vocab id
def SetVocab(v):
    n = 2
    for k in sorted(v.keys()):
        v[k] = n
        n += 1
    # add the begin/end samples
    v['<s>'] = 0
    v['</s>'] = 1
    return v


# trans txt corpus to id corpus
def CorpusToID(fread, fwrite, v, unk='<UNK>', skipwords = 0):
    with open(fread, 'rt') as f1, open(fwrite, 'wt') as f2:
        for line in f1:
            aid = []
            for w in line.upper().split()[skipwords:]:
                if w in v:
                    aid.append(str(v[w]))
                elif unk in v:
                    aid.append(str(v[unk]))
                else:
                    print('[CorpusToID]: {} is not in vocab'.format(w))
            f2.write(' '.join(aid) + '\n')


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
    vlist = sorted(v.items(), key=lambda d: d[1])
    for w, wid in vlist:
        f.write('{}\t{}\n'.format(wid, w))
    f.close()


# read vocabulary
def ReadVocab(fname):
    v = dict()
    f = open(fname, 'rt')
    for line in f:
        a = line.split()
        w = a[1].upper()
        if w in v:
            print('same word with multiple ids (w={})'.format(w))
        else:
            v[w] = int(a[0])
    f.close()
    return v


# Vocab to List:
def VocabToList(v):
    vlist = [0]*len(v)
    for w in v.keys():
        vlist[v[w]] = w
    return vlist


# trans nbest list to id files
def GetNbest(ifile, ofile, v, unk='<UNK>'):
    print('[nbest] ' + ifile + ' -> ' + ofile)
    fin = open(ifile, 'rt')
    fout = open(ofile, 'wt')
    for line in fin:
        a = line.upper().split()
        fout.write(a[0] + ' ')  # write label
        for w in a[1:]:
            nid = v[unk]
            if w in v:
                nid = v[w]
            fout.write('{} '.format(nid))
        fout.write('\n')
    fin.close()
    fout.close()


def Nbest_rmlabel(ifile, ofile):
    with open(ifile) as f1, open(ofile, 'wt') as f2:
        for line in f1:
            a = line.lower().split()
            f2.write(' '.join(a[1:]) + '\n')


def NbestToID(fnbest, fout, v, unk='<UNK>'):
    CorpusToID(fnbest, fout, v, unk, 1)


class model:
    def __init__(self, bindir, workdir):
        self.bindir = wb.folder(bindir)
        self.workdir = wb.folder(workdir)
        wb.mkdir(workdir)

    # prepare corpus
    def prepare(self, train, valid, test, nbest=''):
        v = dict()
        GetVocab(train, v)
        GetVocab(valid, v)
        GetVocab(test, v)
        SetVocab(v)
        WriteVocab(self.workdir + 'vocab', v)

        CorpusToID(train, self.workdir + 'train.no', v)
        CorpusToID(valid, self.workdir + 'valid.no', v)
        CorpusToID(test, self.workdir + 'test.no', v)

        if os.path.exists(nbest):
            GetNbest(nbest, self.workdir + 'nbest.no', v)

    def train(self, config):
        if config.find('-train') == -1:
            config += ' -train {}train.no'.format(self.workdir)
        if config.find('-valid') == -1:
            config += ' -valid {}valid.no'.format(self.workdir)
        cmd = self.bindir + 'rnnlm ' + config
        os.system(cmd)

    # the input is id files
    def ppl_id(self, read_model, test_id):
        cmd = self.bindir + 'rnnlm '
        cmd += ' -rnnlm {} -test {} '.format(read_model, test_id)
        s = os.popen(cmd).read()
        print(s)
        return float(s[s.find('PPL'):].split()[2])

    def ppl(self, read_model, test_txt):
        test_no = self.workdir + os.path.split(test_txt)[-1] + '.pplid'
        CorpusToID(test_txt, test_no, ReadVocab(self.workdir + 'vocab'))
        return self.ppl_id(read_model, test_no)

    # the input nbest is id files
    def rescore_id(self, read_model, read_nbest, write_lmscore):
        write_temp = self.workdir + os.path.split(read_nbest)[1] + '.debug'
        cmd = self.bindir + 'rnnlm '
        cmd += '-rnnlm {} -test {} -nbest -debug 2 > {}'.format(read_model, read_nbest, write_temp)
        os.system(cmd)
        with open(write_temp) as f1, open(read_nbest) as f2, open(write_lmscore, 'wt') as f3:
            for i in range(4):
                f1.readline()
            for line1,line2 in zip(f1, f2):
                label = line2.split()[0]
                score = -float(line1.split()[0]) / math.log10(math.e)
                f3.write('{} {}\n'.format(label, score))

    def rescore(self, read_model, read_nbest_txt, write_lmscore):
        read_nbest_id = self.workdir + os.path.split(read_nbest_txt)[-1] + '.pplid'
        GetNbest(read_nbest_txt, read_nbest_id, ReadVocab(self.workdir + 'vocab'))
        self.rescore_id(read_model, read_nbest_id, write_lmscore)