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

    run_times = range(3, 10)   # for multiple run

    
    bindir = '../../tools/trf/bin/'
    workdir = 'trflm/'
    fres = wb.FRes('models_ppl.txt')
    model = trf.model(bindir, workdir)

    class_num = 200
    train = workdir + 'train.id'
    valid = workdir + 'valid.id'
    test = workdir + 'test.id'
    vocab = workdir + 'vocab_c{}.list'.format(class_num)
    thread = 8

    ais_chain = 10
    ais_inter = 20000

    if '-wer' in sys.argv:
        res_list = []
        for runnum in run_times:
            name = 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run0.ais{}_{}.run{}'.format(ais_chain, ais_inter, runnum)
            res = fres.Get(name)[1:]
            if run_times.index(runnum) == 0:
                res_list = [[] for i in range(len(res))]
            for i in range(len(res)):
                res_list[i].append(res[i])
        name = 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run0.ais{}_{}.avg'.format(ais_chain, ais_inter)
        head = fres.GetHead()[1:]
        for i in range(len(head)):
            mean = np.mean(res_list[i])
            std = np.std(res_list[i])
            fres.Add(name, [head[i]], ['{:.2f}+{:.2f}'.format(mean, std)])

    if '-ais' in sys.argv:
        for runnum in run_times:
            write_model = workdir + 'trf_c200_g4_w_c_ws_cs_wsh_csh_tied.run0'

            [read_nbest, read_templ, read_acscore] = data()[3:6]
            write_templ_id = workdir + os.path.split(read_templ)[1] + '.id'
            v = trf.ReadVocab(vocab)
            trf.NbestToID(read_templ, write_templ_id, v)

            # run asi to calculate the normalization constnts of models
            ais_model = '{}.ais{}_{}.run{}.model'.format(write_model, ais_chain, ais_inter, runnum)
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
            name = os.path.splitext(os.path.split(ais_model)[-1])[0]
            fres.AddLL(name, LL, id_data)
            fres.AddWER(name, wer)
            fres.Add(name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])


if __name__ == '__main__':
    main()
