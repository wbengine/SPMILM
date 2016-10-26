import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import trf


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


def log_add(a,b):
    if a>b:
        return a + np.log(1 + np.exp(b-a))
    else:
        return b + np.log(np.exp(a-b) + 1)
def log_sub(a,b):
    if a>b:
        return a + np.log(abs(1 - np.exp(b-a)))
    else:
        return b + np.log(abs(np.exp(a-b) - 1))
def log_sum(a):
    s = a[0]
    for i in a[1:]:
        s = log_add(s, i)
    return s
def log_mean(a):
    return log_sum(a) - np.log(len(a))
def log_var(a):
    m = log_mean(a)
    b = []
    for i in a:
        b.append(log_sub(i, m) * 2)
    return log_mean(b)

def mat_mean(mat):
    m = []
    for a in np.array(mat):
        m.append(np.log(np.mean(np.exp(a))))
    return m


def mat_var(mat):
    var = []
    for a in np.array(mat):
        var.append(np.log(np.var(np.exp(a))))
    return var


def main():
    if len(sys.argv) == 1:
        print('\"python run.py -train\" train LSTM\n',
              '\"python run.py -rescore\" rescore nbest\n',
              '\"python run.py -wer\" compute WER'
              )

    run_times = range(0, 1)   # for multiple run

    
    bindir = '../../tools/trf/bin/'
    workdir = 'trflm/'
    fres = wb.FRes('models_ppl.txt')
    model = trf.model(bindir, workdir)

    class_num = 200
    train = workdir + 'train.id'
    valid = workdir + 'valid.id'
    test = workdir + 'test.id'
    vocab = workdir + 'vocab_c{}.list'.format(class_num)
    thread = 18

    ais_chain = 10
    ais_inter = 200000

    if '-wer' in sys.argv:
        res_list = []
        for runnum in run_times:
            name = 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run{}.ais{}_{}'.format(runnum, ais_chain, ais_inter)
            res = fres.Get(name)[1:]
            if run_times.index(runnum) == 0:
                res_list = [[] for i in range(len(res))]
            for i in range(len(res)):
                res_list[i].append(res[i])
        name = 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.runavg.ais{}_{}'.format(ais_chain, ais_inter)
        head = fres.GetHead()[1:]
        for i in range(len(head)):
            mean = np.mean(res_list[i])
            std = np.std(res_list[i])
            fres.Add(name, [head[i]], ['{:.2f}+{:.2f}'.format(mean, std)])

    if '-ais' in sys.argv:
        for runnum in run_times:
            write_model = workdir + 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run{}'.format(runnum)

            [read_nbest, read_templ, read_acscore] = data()[3:6]
            write_templ_id = workdir + os.path.split(read_templ)[1] + '.id'
            v = trf.ReadVocab(vocab)
            trf.NbestToID(read_templ, write_templ_id, v)

            # run asi to calculate the normalization constnts of models
            ais_model = '{}.ais{}_{}'.format(write_model, ais_chain, ais_inter)
            if not os.path.exists(ais_model + '.model'):
                config = ' -vocab {0} -read {1}.model -write {2}.model -log {2}.log'.format(vocab, write_model, ais_model)
                config += ' -norm-method AIS -AIS-chain {} -AIS-inter {} -thread {} '.format(ais_chain, ais_inter, thread)
                config += ' -norm-len-max {} '.format(trf.FileMaxLen(read_nbest)-1)  # just compute the needed length
                model.use(config)

            # rescore and compute wer
            write_lmscore = ais_model + '.lmscore'
            config = ' -vocab {} -read {}.model'.format(vocab, ais_model)
            config += ' -nbest {} -test {} '.format(read_nbest, write_templ_id)  # calculate the ppl of test set
            config += ' -lmscore {} '.format(write_lmscore)
            LL_templ = model.use(config, False)
            PPL_templ = wb.LL2PPL(-LL_templ, write_templ_id)
            [wer, lmscale, acscale] = wb.TuneWER(read_nbest, read_templ,
                                                 write_lmscore, read_acscore, np.linspace(0.1, 0.9, 9))
            # calculate the LL of train/valid/test
            LL = [0]*3
            id_data = [train, valid, test]  # are id files
            for i in range(3):
                config = ' -vocab {} -read {}.model -test {} '.format(vocab, ais_model, id_data[i])
                LL[i] = model.use(config, False)

            # write to res file
            name = os.path.split(ais_model)[-1]
            fres.AddLL(name, LL, id_data)
            fres.AddWER(name, wer)
            fres.Add(name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])

    if '-cmp' in sys.argv:
        # compare the variance of exp(logz) with the variance of AIS weight
        # Load the logz of 10 independent runs
        multi_run = 10
        logzs = []
        for i in range(multi_run):
            logz = trf.LoadLogz(workdir + 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run0.ais10_20000.run{}.model'.format(i))
            logzs.append(logz[0:33])
        mat_logzs = np.matrix(logzs).T

        # Load the weight of each length
        logws = []
        with open(workdir + 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run0.ais10_20000.log') as f:
            for line in f:
                idx = line.find('logw=')
                if idx != -1:
                    a = line[idx:].split()[1:]
                    logws.append([float(i) for i in a])
        mat_logws = np.matrix(logws)

        w_var = mat_var(mat_logws)
        z_var = mat_var(mat_logzs)

        for i in range(len(w_var)):
            rate = np.exp(w_var[i] - z_var[i])
            print('len={} w_var={} z_var={} rate={}'.format(i+1, w_var[i], z_var[i], rate))
    if '-cmp2' in sys.argv:
        # compare the logz of AIS and the SAMS
        write_model = workdir + 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run0'
        logz_sams = trf.LoadLogz(write_model + '.model')
        logz_ais = trf.LoadLogz('{}.ais{}_{}.model'.format(write_model, ais_chain, ais_inter))
        logz_ais1 = trf.LoadLogz('{}.ais10_20000.run0.model'.format(write_model))
        plt.figure()
        plt.plot(logz_sams[0:33], 'r-', label='sams')
        plt.plot(logz_ais1[0:33], 'g--', label='ais 10-20K')
        plt.plot(logz_ais[0:33], 'b--', label='ais 10-200K')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
