import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import trf
import rnn
import lstm

# revise this function to config the dataset used to train different model
def data():
    # ptb + wsj93 experiments
    return data_ptb() + data_wsj92nbest()


def data_verfy(paths):
    for w in paths:
        if not os.path.isfile(w):
            print('[ERROR] no such file: ' + w)
    return paths


def data_ptb():
    root = './data/ptb/'
    train = root + 'ptb.train.txt'
    valid = root + 'ptb.valid.txt'
    test = root + 'ptb.test.txt'
    return data_verfy([train, valid, test])


def data_wsj92nbest():
    root = './data/WSJ92-test-data/'
    nbest = root + '1000best.sent'
    trans = root + 'transcript.txt'
    ac = root + '1000best.acscore'
    lm = root + '1000best.lmscore'
    return data_verfy([nbest, trans, ac, lm])


def wer_all(lmpaths, lmtypes, outlog):
    fres = wb.FRes(outlog, True)
    fres.Clean()

    [read_nbest, read_trans, read_acscore] = data()[3:6]
    lmscale_vec = np.linspace(0.1, 0.9, 9)
    weight_vec = np.linspace(0.5, 0.5, 1)

    for type in lmtypes:
        exist_multiple_run = False
        a = type.split('+')
        for lm in a:
            if lmpaths[lm].find('<run>') != -1:
                exist_multiple_run = True
                break

        run_vec = [0]
        run_name = type
        if exist_multiple_run:
            run_vec = range(0, 10)
            run_name = type + ':<run>'

        for run in run_vec:
            run_str = 'run{}'.format(run)
            name = run_name.replace('<run>', run_str)
            opt_wer_vec = [100, 1.0, 1.0]
            opt_weight = 1.0

            if len(a) == 1:
                lmscore = wb.LoadScore(lmpaths[a[0]].replace('<run>', run_str))
                opt_wer_vec = wb.TuneWER(read_nbest, read_trans,
                                                     lmscore, read_acscore, lmscale_vec)
                opt_weight = 1.0
            else:
                lmscore1 = np.array(wb.LoadScore(lmpaths[a[0]].replace('<run>', run_str)))
                lmscore2 = np.array(wb.LoadScore(lmpaths[a[1]].replace('<run>', run_str)))

                for w in weight_vec:
                    lmscore = w*lmscore1 + (1-w)*lmscore2
                    [wer, lmscale, acscale] = wb.TuneWER(read_nbest, read_trans,
                                                     lmscore, read_acscore, lmscale_vec)
                    if wer < opt_wer_vec[0]:
                        opt_wer_vec = [wer, lmscale, acscale]
                        opt_weight = w

            fres.Add(name, ['wer', 'lmscale', 'acscale', 'weight'], opt_wer_vec + [opt_weight])



if __name__ == '__main__':
    lmpaths = {'KN5': 'ngramlm/5gram.lmscore',
               'RNN': 'rnnlm/h250_c1_bptt5.run0.lmscore',
               'LSTM':'lstmlm/h250_dropout0_epoch10.run0.lmscore',
               'TRF': 'trflm/trf_c200_g4_w_c_ws_cs_wsh_csh_tied.<run>.lmscore'}
    lmtypes = ['KN5', 'RNN', 'LSTM', 'TRF', 'RNN+KN5', 'RNN+TRF', 'LSTM+KN5', 'LSTM+TRF']
    outlog = 'wer.log'

    if not os.path.exists(outlog):
        wer_all(lmpaths, lmtypes, outlog)

    fres = wb.FRes(outlog, True)

    lmwers = dict()
    with open(outlog, 'rt') as f:
        f.readline()
        for a in [line.split() for line in f]:
            if a[0].find('[all]') != -1:
                break
            type = a[0].split(':')[0]
            wer_vec = lmwers.setdefault(type, [])
            wer_vec.append(float(a[1]))

    for type in lmtypes:
        wer_vec = lmwers[type]
        wer_mean = np.mean(wer_vec)
        wer_std = np.std(wer_vec)
        fres.Add(type + '[all]', ['wer'], ['{:.3f}+{:.3f}'.format(wer_mean, wer_std)])


