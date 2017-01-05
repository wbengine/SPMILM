import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import ngram


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


def main():
    print(sys.argv)
    if len(sys.argv) == 1:
        print('\"python run_ngram.py -train\" train \n',
              '\"python run_ngram.py -rescore\" rescore nbest\n',
              '\"python run_ngram.py -wer\" compute WER'
              )


    bindir = '../../tools/srilm/'
    fres = wb.FRes('result.txt')  # the result file
    order_reg = [5]

    for tsize in [1, 2, 4]:
        tskdir = '{}/'.format(tsize)
        workdir = tskdir + 'ngramlm/'
        model = ngram.model(bindir, workdir)

        for order in order_reg:
            write_model = workdir + '{}gram.lm'.format(order)
            write_name = '{}:KN{}'.format(tsize, order)

            print(write_model)

            if '-train' in sys.argv or '-all' in sys.argv:
                if order_reg.index(order) == 0:
                    model.prepare(data(tskdir)[0], data(tskdir)[1], data(tskdir)[2])
                model.train(order, write_model)

            if '-test' in sys.argv or '-all' in sys.argv:
                PPL = [0]*3
                PPL[0] = model.ppl(write_model, order, data(tskdir)[0])
                PPL[1] = model.ppl(write_model, order, data(tskdir)[1])
                PPL[2] = model.ppl(write_model, order, data(tskdir)[2])
                fres.AddPPL(write_name, PPL, data(tskdir)[0:3])

            if '-rescore' in sys.argv or '-all' in sys.argv:
                model.rescore(write_model, order, data(tskdir)[3], write_model[0:-3] + '.lmscore')

            if '-wer' in sys.argv or '-all' in sys.argv:
                [nbest, templ] = data(tskdir)[3:5]
                lmscore = wb.LoadScore(write_model[0:-3] + '.lmscore')
                acscore = wb.LoadScore(data(tskdir)[5])

                [wer, lmscale, acscale] = wb.TuneWER(nbest, templ, lmscore, acscore, np.linspace(0.1, 0.9, 9))
                print('wer={} lmscale={} acscale={}'.format(wer, lmscale, acscale))
                fres.AddWER(write_name, wer)

                trans_txt = workdir + os.path.split(templ)[-1] + '.txt'
                wb.file_rmlabel(templ, trans_txt)
                PPL_temp = model.ppl(write_model, order, trans_txt)
                LL_temp = -wb.PPL2LL(PPL_temp, trans_txt)
                fres.Add(write_name, ['LL-wsj', 'PPL-wsj'], [LL_temp, PPL_temp])


if __name__ == '__main__':
    main()
