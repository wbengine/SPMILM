import os
import sys
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
    return v


# trans txt corpus to id corpus
def CorpusToID(fread, fwrite, v, unk='<UNK>'):
    print('[w2id] ' + fread + ' -> ' + fwrite)
    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.upper().split()
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
    v1 = [''] * (len(v) + 2)
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
    f.write('<s>\n</s>\n')
    for w, wid in vlist:
        f.write('{}\t{}\n'.format(wid, w))
    f.close()


# read vocabulary
def ReadVocab(fname):
    v = dict()
    f = open(fname, 'rt')
    f.readline()
    f.readline()
    for line in f:
        a = line.split()
        v[a[1].upper()] = int(a[0])
    f.close()
    return v


# trans nbest list to id files
def GetNbest(ifile, ofile, v, unk='<UNK>'):
    print('[nbest] ' + ifile + ' -> ' + ofile)
    fin = open(ifile, 'rt')
    fout = open(ofile, 'wt')
    for line in fin:
        a = line.upper().split()
        for w in a[1:]:
            nid = 0
            if w in v:
                nid = v[w]
            elif unk in v:
                nid = v[unk]
            else:
                print('[error] on word in vocabulary ' + w)
            fout.write('{} '.format(nid))
        fout.write('\n')
    fin.close()
    fout.close()
    
# trans the debug 2 output of SRILM to sentence score
def Debug2SentScore(fdbg, fscore):
    with open(fdbg, 'rt') as f1, open(fscore, 'wt') as f2:
        score = []
        for line in f1:
            if 'logprob=' not in line:
                continue
            score.append(-float(line[line.find('logprob='):].split()[1]))
        for i in range(len(score)-1):
            f2.write('sent={}\t{}\n'.format(i, score[i]))


class model:
    def __init__(self,  bindir, workdir):
        self.workdir = wb.folder(workdir)
        self.bindir = wb.folder(bindir)
        wb.mkdir(workdir)

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

    def train(self, order, write_model, output_res='', discount='-kndiscount', cutoff=[]):
        write_count = self.workdir + os.path.split(write_model)[1] + '.count'

        cmd = self.bindir + 'ngram-count '
        cmd += ' -text {0}train.no -vocab {0}vocab'.format(self.workdir)
        cmd += ' -order {} -write {} '.format(order, write_count)
        for i in range(len(cutoff)):
            cmd += ' -gt{}min {}'.format(i+1, cutoff[i])
        os.system(cmd)

        cmd = self.bindir + 'ngram-count '
        cmd += ' -vocab {}vocab'.format(self.workdir)
        cmd += ' -read {}'.format(write_count)
        cmd += ' -order {} -lm {} '.format(order, write_model)
        cmd += discount + ' -interpolate -gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0'
        os.system(cmd)

        # get ppl
        if output_res != '':
            PPL = [0] * 3
            testno = [self.workdir + i + '.no' for i in ['train', 'valid', 'test']]
            for i in range(3):
                cmd = self.bindir + 'ngram -order {} -lm {} -ppl {}'.format(order, write_model, testno[i])
                res = os.popen(cmd).read()
                PPL[i] = float(res[res.find('ppl='):].split()[1])
            res_file = wb.FRes(output_res)
            res_file.AddPPL('KN{}'.format(order), PPL, testno)

    def ppl(self, lm, order, test):
        [a, b] = os.path.split(test)
        write_test = self.workdir + b + '.pplid'

        v = ReadVocab(self.workdir + 'vocab')
        CorpusToID(test, write_test, v)

        cmd = self.bindir + 'ngram -order {} -lm {} -ppl {}'.format(order, lm, write_test)
        res = os.popen(cmd).read()
        print(res)
        return float(res[res.find('ppl='):].split()[1])
    
    def rescore(self, lm, order, nbest, lmscore):
        write_nbest_no = self.workdir + 'nbest.no'
        write_temp = self.workdir + 'nbest.no.debug'
        
        v = ReadVocab(self.workdir + 'vocab')
        GetNbest(nbest, write_nbest_no, v)
        
        cmd = self.bindir + 'ngram -lm {} -order {} -ppl {} -debug 2 > {}'.format(lm, order, write_nbest_no, write_temp)
        os.system(cmd)
        Debug2SentScore(write_temp, lmscore)
        
        
        
        
        
        
        
