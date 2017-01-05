import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd() + '/../../tools/')
import lstm
import wb
import wer


def rescore_all(workdir, nbestdir, config):
    for tsk in ['nbestlist_{}_{}'.format(a, b) for a in ['dt05', 'et05'] for b in ['real', 'simu']]:
        print('process ' + tsk)
        nbest_txt = nbestdir + tsk + '/words_text'
        outdir = workdir + nbestdir.split('/')[-2] + '/' + tsk + '/'
        wb.mkdir(outdir)

        write_lmscore = outdir + 'lmwt.lstm'
        lstm.rescore(workdir, nbest_txt, write_lmscore, config)


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 1:
        print(
        ' \"python run.py -train\" train LSTM\n \"python run.py -rescore\" rescore nbest\n \"python run.py -wer\" compute WER')

    absdir = os.getcwd() + '/'
    train = absdir + 'data/train'
    valid = absdir + 'data/valid'
    nbestdir = absdir + 'data/nbest/nbest_mvdr_single_heq_multi/'
    workdir = absdir + 'lstmlm/'
    wb.mkdir(workdir)
    os.chdir('../../tools/lstm/')

    config = '-hidden 500 -epoch 10 -dropout 0 -gpu 2'

    if '-train' in sys.argv:
        lstm.train(workdir, train, valid, valid, config)
    if '-test' in sys.argv:
        lstm.ppl(workdir, train, config)
        lstm.ppl(workdir, valid, config)
    if '-rescore' in sys.argv:
        rescore_all(workdir, nbestdir, config)
    if '-wer' in sys.argv:
        lmpaths = {'KN5': nbestdir + '<tsk>/lmwt.lmonly',
                   'RNN': nbestdir + '<tsk>/lmwt.rnn',
                   'LSTM': workdir + nbestdir.split('/')[-2] + '/<tsk>/lmwt.lstm',
                   'TRF': '/home/ozj/NAS_workspace/wangb/Experiments/ChiME4/lmscore/' + nbestdir.split('/')[
                       -2] + '/<tsk>/lmwt.trf'}
        # 'TRF': nbestdir + '<tsk>/lmwt.trf'}
        lmtypes = ['LSTM', 'KN5', 'RNN', 'TRF', 'RNN+KN5', 'LSTM+KN5', 'RNN+TRF', 'LSTM+TRF']
        # lmtypes = ['LSTM', 'LSTM+TRF']
        wer_workdir = absdir + 'wer/' + nbestdir.split('/')[-2] + '/'
        print('wer_workdir = ' + wer_workdir)
        wer.wer_all(wer_workdir, nbestdir, lmpaths, lmtypes)
        config = wer.wer_tune(wer_workdir)
        wer.wer_print(wer_workdir, config)
