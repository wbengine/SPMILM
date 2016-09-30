import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import trf
import wer


def process_nbest(fread, fwrite):
    nEmptySentNum = 0
    with open(fread, 'rt') as f1, open(fwrite, 'wt') as f2:
        for a in [line.split() for line in f1]:
            if len(a) == 1:
                nEmptySentNum += 1
                a.append('<UNK>')
            f2.write(' '.join(a) + '\n')
    print('[nbest] empty sentence num = {}'.format(nEmptySentNum))


def main():
    if len(sys.argv) == 1:
        print('\"python run.py -train\" train LSTM\n',
              '\"python run.py -rescore\" rescore nbest\n',
              '\"python run.py -wer\" compute WER'
              )

    bindir = '../../tools/trf/bin/'
    workdir = 'trflm/'
    fres = wb.FRes('models_ppl.txt')
    model = trf.model(bindir, workdir)

    nbest_root = 'data/nbest/'
    nbest_type_list = ['nbest_mvdr_single_heq_multi']

    class_num = 200
    train = workdir + 'train.id'
    valid = workdir + 'valid.id'
    test = workdir + 'test.id'
    vocab = workdir + 'vocab_c{}.list'.format(class_num)
    order = 4
    feat = 'g4_w_c_ws_cs_wsh_csh_tied.fs'
    #feat = 'g4_w_c_ws_cs_wsh_csh.fs'
    maxlen = 0
    tmax = 50000
    t0 = 10000
    minibatch = 100
    gamma_lambda = '3000,0'
    gamma_zeta = '0,0.6'
    reg = 1e-6
    thread = 8

    write_model = workdir + 'trf_c{}_{}'.format(class_num, feat[0:-3])
    if '-train' in sys.argv or '-all' in sys.argv:
        config = '-vocab {} -train {} -valid {} -test {} '.format(vocab, train, valid, test)
        config += ' -order {} -feat {} '.format(order, feat)
        config += ' -len {} '.format(maxlen)
        config += ' -write {0}.model -log {0}.log '.format(write_model)
        config += ' -t0 {} -iter {}'.format(t0, tmax)
        config += ' -gamma-lambda {} -gamma-zeta {}'.format(gamma_lambda, gamma_zeta)
        config += ' -L2 {} '.format(reg)
        config += ' -mini-batch {} '.format(minibatch)
        config += ' -thread {} '.format(thread)
        config += ' -print-per-iter 10 '
        config += ' -write-at-iter [{}:10000:{}]'.format(tmax-30000, tmax)  # output the intermediate models
        model.prepare('data/train', 'data/valid', 'data/valid', class_num)
        model.train(config)
    if '-plot' in sys.argv:
        baseline = fres.Get('KN5')
        trf.PlotLog([write_model], [baseline])
    if '-rescore' in sys.argv or '-all' in sys.argv:
        for nbest_type in nbest_type_list:
            nbest_dir = nbest_root + nbest_type + '/'
            for tsk in ['nbestlist_{}_{}'.format(a, b) for a in ['dt05', 'et05'] for b in ['real', 'simu']]:
                write_dir = workdir + nbest_type + '/' + tsk + '/'
                wb.mkdir(write_dir)
                print('{} : {}'.format(nbest_type, tsk))
                print('  write -> {}'.format(write_dir))
                write_lmscore = write_dir + os.path.split(write_model)[-1]
                # fill the empty lines
                process_nbest(nbest_dir + tsk + '/words_text', write_lmscore + '.nbest')

                config = ' -vocab {} '.format(vocab)
                config += ' -read {}.model '.format(write_model)
                config += ' -nbest {} '.format(write_lmscore + '.nbest')
                config += ' -lmscore {0}.lmscore -lmscore-test-id {0}.test-id '.format(write_lmscore)
                model.use(config)
    if '-wer' in sys.argv or '-all' in sys.argv:
        for nbest_type in nbest_type_list:
            nbest_dir = nbest_root + nbest_type + '/'
            lmpaths = {'KN5': nbest_dir + '<tsk>/lmwt.lmonly',
                       'RNN': nbest_dir + '<tsk>/lmwt.rnn',
                       'LSTM': 'lstm/' + nbest_type + '/<tsk>/lmwt.lstm',
                       'TRF':  workdir + nbest_type + '/<tsk>/' + os.path.split(write_model)[-1] + '.lmscore'}
            # 'TRF': nbestdir + '<tsk>/lmwt.trf'}
            # lmtypes = ['LSTM', 'KN5', 'RNN', 'TRF', 'RNN+KN5', 'LSTM+KN5', 'RNN+TRF', 'LSTM+TRF']
            lmtypes = ['TRF','RNN','KN5', 'RNN+TRF']
            wer_workdir = 'wer/' + nbest_type + '/'
            print('wer_workdir = ' + wer_workdir)
            wer.wer_all(wer_workdir, nbest_dir, lmpaths, lmtypes)
            config = wer.wer_tune(wer_workdir)
            wer.wer_print(wer_workdir, config)

if __name__ == '__main__':
    main()
