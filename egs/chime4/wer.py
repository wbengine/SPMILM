import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb


def nbest_rmUNK(read_nbest, write_nbest):
    f = open(read_nbest, 'rt')
    fo = open(write_nbest, 'wt')
    for line in f:
        fo.write(line.replace('<UNK>', ''))
    f.close()
    fo.close()


# such as 
# lmpaths = {'KN5': nbestdir + '<tsk>/lmwt.lmonly',
#	       'RNN': nbestdir +'<tsk>/lmwt.rnn',
#	       'LSTM': workdir + '<tsk>/lmwt.lstm'}
# lmtypes = ['KN5', 'RNN', 'RNN+KN5']
def wer_all(workdir, nbestdir, lmpaths, lmtypes):
    wb.mkdir(workdir)
    # calculate the wer for each task, each lmscale, each combination
    for tsk in ['nbestlist_{}_{}'.format(a, b) for a in ['dt05', 'et05'] for b in ['real', 'simu']]:
        print(tsk)
        wb.mkdir(workdir + tsk)
        fwer = open(workdir + tsk + '/wer.txt', 'wt')

        read_nbest_txt = nbestdir + tsk + '/words_text'
        read_transcript = nbestdir + tsk + '/text'
        read_acscore = nbestdir + tsk + '/acwt'
        read_gfscore = nbestdir + tsk + '/lmwt.nolm'

        # remove the <UNK> in nbest
        read_nbest_rmunk = workdir + tsk + '/words_text_rmunk'
        nbest_rmUNK(read_nbest_txt, read_nbest_rmunk)
        # load score
        acscore = np.array(wb.LoadScore(read_acscore))
        gfscore = np.array(wb.LoadScore(read_gfscore))
        # load label
        score_label = wb.LoadLabel(read_acscore)

        # lm config

        for lmtype in lmtypes:
            a = lmtype.split('+')
            if len(a) == 1:
                lmscore = np.array(wb.LoadScore(lmpaths[a[0]].replace('<tsk>', tsk)))
            elif len(a) == 2:
                s1 = wb.LoadScore(lmpaths[a[0]].replace('<tsk>', tsk))
                s2 = wb.LoadScore(lmpaths[a[1]].replace('<tsk>', tsk))
                lmscore = 0.5 * np.array(s1) + 0.5 * np.array(s2)

            # write lmscore
            wb.WriteScore(workdir + tsk + '/' + lmtype + '.lmscore', lmscore, score_label)

            for lmscale in np.linspace(9, 15, 7):
                write_best = workdir + tsk + '/{}_lmscale={}.best'.format(lmtype, lmscale)
                wb.GetBest(read_nbest_rmunk, (acscore + lmscale * (lmscore + gfscore)).tolist(), write_best)
                [err, num, wer] = wb.CmpWER(write_best, read_transcript)
                os.remove(write_best)
                s = '{} wer={:.2f} err={} num={} lmscale={}'.format(lmtype, wer, err, num, lmscale)
                print('  ' + s)
                fwer.write(s + '\n')
                fwer.flush()

        fwer.close()


def wer_tune(workdir):
    config = {}
    nLine = 0
    with open(workdir + 'nbestlist_dt05_real/wer.txt') as f1, open(workdir + 'nbestlist_dt05_simu/wer.txt') as f2, open(
                    workdir + 'dt05_real_simu_wer.txt', 'wt') as f3:
        for linea, lineb in zip(f1, f2):
            a = linea.split()
            b = lineb.split()
            if (a[0] != b[0] or a[4:] != b[4:]):
                print('[ERROR] wer_tune : two files are not match')
                print(linea)
                print(lineb)
                return

            weight = float(a[4].split('=')[1])
            totale = int(a[2].split('=')[1]) + int(b[2].split('=')[1])
            totalw = int(a[3].split('=')[1]) + int(b[3].split('=')[1])
            wer = 100.0 * totale / totalw
            f3.write('{} wer={:.2f} err={} num={} lmscale={}\n'.format(a[0], wer, totale, totalw, weight))

            x = config.setdefault(a[0], dict())
            oldwer = x.setdefault('wer', 100)
            if wer < oldwer:
                x['wer'] = wer
                x['line'] = nLine

            nLine += 1
    return config


def ExactVaue(s, label):
    a = s.split()
    l = len(label)
    for w in a:
        if w[0:l] == label:
            return w[l + 1:]
    return s


def wer_print(workdir, config):
    fresult = open(workdir + 'wer_result.txt', 'wt')

    keylist = []
    for tsk in ['nbestlist_{}_{}'.format(a, b) for b in ['real', 'simu'] for a in ['dt05', 'et05']]:
        print(tsk)
        fresult.write(tsk + '\n')
        keylist.append(tsk)
        fwer = open(workdir + tsk + '/wer.txt')
        a = fwer.readlines()
        for key in config.keys():
            x = config[key]
            n = x['line']
            line = a[n][0:-1]  # remove '\n'
            x[tsk] = float(ExactVaue(line, 'wer'))
            x['lmscale'] = float(ExactVaue(line, 'lmscale'))
            print('  ' + a[n][0:-1])
            fresult.write(a[n])

    # print
    s = 'model\tlmscale\t' + '\t'.join([i[10:] for i in keylist])
    print(s)
    fresult.write(s + '\n')
    for key in sorted(config.keys()):
        x = config[key]
        s = '{}\t{}\t{}\t{}\t{}\t{}'.format(key, x['lmscale'], x[keylist[0]], x[keylist[1]], x[keylist[2]],
                                            x[keylist[3]])
        print(s)
        fresult.write(s + '\n')

    fresult.close()


# main
if __name__ == '__main__':
    absdir = os.getcwd() + '/'
    nbestdir = absdir + 'data/NBEST_HEQ/'
    lmpaths = {'KN5': nbestdir + '<tsk>/lmwt.lmonly',
               'RNN': nbestdir + '<tsk>/lmwt.rnn'}
    lmtypes = ['KN5', 'RNN', 'RNN+KN5']
    wer_workdir = absdir + 'wer/'

    # compute the wer for all the dataset
    wer_all(wer_workdir, nbestdir, lmpaths, lmtypes)
    # using dt_real and dt_simu to tune the lmscale
    config = wer_tune(wer_workdir)
    # using the tuned lmscale to get the result WER
    wer_print(wer_workdir, config)
