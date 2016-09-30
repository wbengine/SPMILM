import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import trf


# revise this function to config the dataset used to train different model
def data(tskdir):
    train = tskdir + 'data/train.txt'
    valid = tskdir + 'data/valid.txt'
    test = tskdir + 'data/test.txt'
    return data_verfy([train, valid, test]) + data_wsj92nbest()


def data_verfy(paths):
    for w in paths:
        if not os.path.isfile(w):
            print('[ERROR] no such file: ' + w)
    return paths


def data_wsj92nbest():
    root = './data/WSJ92-test-data/'
    nbest = root + '1000best.sent'
    trans = root + 'transcript.txt'
    ac = root + '1000best.acscore'
    lm = root + '1000best.lmscore'
    return data_verfy([nbest, trans, ac, lm])


def evaulate_trf(model, vocab, read_model, tsize, fres):
    res_name = '{}:'.format(int(tsize)) + os.path.split(read_model)[-1]
    tskdir = '{}/'.format(tsize)

    # rescore
    config = ' -vocab {} '.format(vocab)
    config += ' -read {}.model '.format(read_model)
    config += ' -nbest {} '.format(data(tskdir)[3])
    config += ' -lmscore {0}.lmscore'.format(read_model)
    model.use(config)
    # WER
    [read_nbest, read_templ, read_acscore, read_lmscore] = data(tskdir)[3:7]
    read_lmscore = read_model + '.lmscore'

    [wer, lmscale, acscale] = wb.TuneWER(read_nbest, read_templ,
                                     wb.LoadScore(read_lmscore),
                                     wb.LoadScore(read_acscore), np.linspace(0.1,0.9,9))
    print('wer={:.4f} lmscale={:.2f} acscale={:.2f}'.format(wer, lmscale, acscale))
    # calculate the ppl on wsj test
    templ_txt = model.workdir + os.path.split(read_templ)[-1] + '.rmlabel'
    wb.file_rmlabel(read_templ, templ_txt)
    PPL_templ = model.ppl(vocab, read_model+'.model', templ_txt)
    LL_templ = -wb.PPL2LL(PPL_templ, templ_txt)

    # output the result
    fres.Add(res_name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])
    fres.AddWER(res_name, wer)



def main():
    if len(sys.argv) == 1:
        print('\"python run.py -train\" train LSTM\n',
              '\"python run.py -rescore\" rescore nbest\n',
              '\"python run.py -wer\" compute WER'
              )


    for tsize in [4]:
        bindir = '../../tools/trf/bin/'
        tskdir = '{}/'.format(tsize)
        workdir = tskdir + 'trflm/'

        fres = wb.FRes('result.txt')
        model = trf.model(bindir, workdir)

        class_num = 200
        train = workdir + 'train.id'
        valid = workdir + 'valid.id'
        test = workdir + 'test.id'
        vocab = workdir + 'vocab_c{}.list'.format(class_num)
        order = 4
        feat = 'g4_w_c_ws_cs_wsh_csh_tied.fs'
        #feat = 'g4_w_c_ws_cs_cpw.fs'
        maxlen = 100
        tmax = 50000
        t0 = 2000
        minibatch = 100
        gamma_lambda = '1000,0'
        gamma_zeta = '0,0.6'
        reg = 1e-5
        thread = 8

        write_model = workdir + 'trf_c{}_{}'.format(class_num, feat[0:-3])
        write_name = '{}:{}'.format(tsize, os.path.split(write_model)[1])

        if '-class' in sys.argv:
            # just cluster for each tsks.
            model.prepare(data(tskdir)[0], data(tskdir)[1], data(tskdir)[2], class_num)
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
            config += ' -print-per-iter 10 -write-at-iter 10000:10000:{}'.format(tmax)
            model.prepare(data(tskdir)[0], data(tskdir)[1], data(tskdir)[2], class_num)
            model.train(config)
            # output
            LL = model.get_last_value(write_model + '.log')
            fres.AddLL(write_name, LL, data(tskdir)[0:3])
        if '-plot' in sys.argv:
            baseline = fres.Get('{}:KN5'.format(tsize))
            trf.PlotLog([write_model], [baseline])
        if '-rescore' in sys.argv or '-all' in sys.argv:
            config = ' -vocab {} '.format(vocab)
            config += ' -read {}.model '.format(write_model)
            config += ' -nbest {} '.format(data(tskdir)[3])
            config += ' -lmscore {0}.lmscore -lmscore-test-id {0}.test-id '.format(write_model)
            model.use(config)
        if '-wer' in sys.argv or '-all' in sys.argv:
            [read_nbest, read_templ, read_acscore, read_lmscore] = data(tskdir)[3:7]
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

            # output the result
            fres.Add(write_name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])
            fres.AddWER(write_name, wer)
        if '-inter' in sys.argv:
            # calculate the WER for intermediate models
            for n in np.linspace(10000, 40000, 4):
                inter_model = workdir + 'trf_c{}_{}.n{}'.format(class_num, feat[0:-3], int(n))
                evaulate_trf(model, vocab, inter_model, tsize, fres)

if __name__ == '__main__':
    main()
