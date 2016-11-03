import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import wb
import trf

# plot the loglikelihood from log files
# baseline is [ ['KN5', 1,2,3,4,5,6 ], ['RNN',1,2,3,4,5,6] ]
def PlotLog(name_pack, baseline=[]):
    trf.PlotLog(name_pack, baseline)

class model:
    def __init__(self, bindir, workdir):
        self.bindir = wb.folder(bindir)
        self.workdir = wb.folder(workdir)
        wb.mkdir(workdir)

    def WordCluster(self, fvocab, fid, class_num, write_list):
        if os.path.exists(write_list):
            print('vocab exist, skip: ' + write_list)
            return

        cmd = self.bindir + 'word-cluster -txt {} -num {} -tag-vocab {} '.format(fid, class_num, fid + '.vocab')
        trf.System(cmd)

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
        trf.GetVocab(train, v)
        trf.GetVocab(valid, v)
        # GetVocab(test, v)
        trf.WriteVocab(write_vocab, v)
        trf.CorpusToID(train, write_train, v)
        trf.CorpusToID(valid, write_valid, v)
        trf.CorpusToID(test, write_test, v)

        # get class
        self.WordCluster(write_vocab, write_train, class_num, write_vocab_c)

    def train(self, config):
        cmd = self.bindir + 'hrf-satrain ' + config
        trf.System(cmd)

    def train_ml(self, config):
        cmd = self.bindir + 'hrf-mltrain ' + config
        trf.System(cmd)

    def use(self, config):
        cmd = self.bindir + 'hrf ' + config
        trf.System(cmd)
