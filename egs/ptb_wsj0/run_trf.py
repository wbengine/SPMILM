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

    class_num = 200
    train = workdir + 'train.id'
    valid = workdir + 'valid.id'
    test = workdir + 'test.id'
    vocab = workdir + 'vocab_c{}.list'.format(class_num)
    order = 4
    feat = 'g4_w_c_ws_cs_wsh_csh_tied.fs'
    #feat = 'g4_w_c_ws_cs_cpw.fs'
    maxlen = 0
    tmax = 20000
    t0 = 2000
    minibatch = 100
    gamma_lambda = '3000,0'
    gamma_zeta = '0,0.6'
    reg = 4e-5
    thread = 12

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
        config += ' -write-at-iter [{}:1000:{}]'.format(tmax-5000, tmax)  # output the intermediate models
        model.prepare(data()[0], data()[1], data()[2], class_num)
        model.train(config)
    if '-plot' in sys.argv:
        fres = wb.FRes('models_ppl.txt')
        baseline = fres.Get('KN5')
        trf.PlotLog([write_model], [baseline])
    if '-rescore' in sys.argv or '-all' in sys.argv:
        config = ' -vocab {} '.format(vocab)
        config += ' -read {}.model '.format(write_model)
        config += ' -nbest {} '.format(data()[3])
        config += ' -lmscore {0}.lmscore -lmscore-test-id {0}.test-id '.format(write_model)
        model.use(config)
    if '-wer' in sys.argv or '-all' in sys.argv:
        [read_nbest, read_templ, read_acscore, read_lmscore] = data()[3:7]
        read_lmscore = write_model + '.lmscore'

        [wer, lmscale, acscale] = wb.TuneWER(read_nbest, read_templ,
                                             wb.LoadScore(read_lmscore),
                                             wb.LoadScore(read_acscore), np.linspace(0.1,0.9,9))
        print('wer={:.4f} lmscale={:.2f} acscale={:.2f}'.format(wer, lmscale, acscale))

        # calculate the ppl on wsj test
        write_templ_id = workdir + os.path.split(read_templ)[1] + '.id'
        v = trf.ReadVocab(vocab)
        trf.NbestToID(read_templ, write_templ_id, v)
        config = ' -vocab {} '.format(vocab)
        config += ' -read {}.model '.format(write_model)
        config += ' -test {} '.format(write_templ_id)
        LL_templ = model.use(config)
        PPL_templ = wb.LL2PPL(-LL_templ, write_templ_id)
        LL = model.get_last_value(write_model + '.log')

        # output the result
        name = os.path.split(write_model)[1]
        fres.AddLL(name, LL, data()[0:3])
        fres.Add(name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])
        fres.AddWER(name, wer)
    if '-stat' in sys.argv:
        # calculate the mean and var of wers of the intermediate models
        inte_wer = []
        inte_model = []

        # find model
        for file_name in os.listdir(os.path.split(write_model)[0]):
            file_path = os.path.split(write_model)[0] + os.sep + file_name
            if not os.path.isfile(file_path):
                continue
            if file_name.find(os.path.split(write_model)[1]) == 0 and \
                file_path.split('.')[-1] == 'model' and \
                file_path.split('.')[-2][0] == 'n':
                inte_model.append(file_path)

        # compute wer
        flog = open(workdir + 'inte_model_wer.log', 'wt')
        for file_path in sorted(inte_model):
            print(file_path)
            t = int(file_path.split('.')[-2][1:])

            # lmscore
            write_lmscore = os.path.splitext(file_path)[0] + '.lmscore'
            config = ' -vocab {} '.format(vocab)
            config += ' -read {} '.format(file_path)
            config += ' -nbest {} '.format(data()[3])
            config += ' -lmscore {0} '.format(write_lmscore)
            model.use(config, False)
            # wer
            [wer, lmscale, acscale] = wb.TuneWER(data()[3], data()[4],
                                             wb.LoadScore(write_lmscore),
                                             wb.LoadScore(data()[5]), np.linspace(0.1, 0.9, 9))
            print('t={} wer={}'.format(t, wer))
            flog.write('{} \t wer={}\n'.format(file_path, wer))
            inte_wer.append([t, wer])
        flog.close()

        # plot wer
        inte_wer = sorted(inte_wer, key=lambda d: d[0])
        t_list = [i[0] for i in inte_wer]
        wer_list = [i[1] for i in inte_wer]
        wer_mean = np.mean(wer_list[-20:])
        wer_std = np.std(wer_list[-20:])
        print('wer_mean={}  wer_std={}'.format(wer_mean, wer_std))

        plt.figure()
        plt.plot(t_list, wer_list)
        plt.xlabel('t')
        plt.ylabel('wer')
        plt.show()
    if '-ais' in sys.argv:
        [read_nbest, read_templ, read_acscore] = data()[3:6]
        write_templ_id = workdir + os.path.split(read_templ)[1] + '.id'
        v = trf.ReadVocab(vocab)
        trf.NbestToID(read_templ, write_templ_id, v)

        # run asi to calculate the normalization constnts of models
        ais_chain = 10
        ais_inter = 10000
        ais_model = '{}.ais{}_{}.model'.format(write_model, ais_chain, ais_inter)
        if not os.path.exists(ais_model):
            config = ' -vocab {0} -read {1}.model -write {2}'.format(vocab, write_model, ais_model)
            config += ' -norm-method AIS -AIS-chain {} -AIS-inter {} -thread {} '.format(ais_chain, ais_inter, thread)
            config += ' -norm-len-max {} '.format(trf.FileMaxLen(read_nbest)-1)  # just compute the needed length
            model.use(config)

        # rescore and compute wer
        write_lmscore = os.path.splitext(ais_model)[0] + '.lmscore'
        config = ' -vocab {} -read {}'.format(vocab, ais_model)
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
            config = ' -vocab {} -read {} -test {} '.format(vocab, ais_model, id_data[i])
            LL[i] = model.use(config, False)

        # write to res file
        name = os.path.split(write_model)[1]+":AIS{}-{}".format(ais_chain, ais_inter)
        fres.AddLL(name, LL, id_data)
        fres.AddWER(name, wer)
        fres.Add(name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])

if __name__ == '__main__':
    main()
