import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import ngram


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
    print(sys.argv)
    if len(sys.argv) == 1:
        print('\"python run_ngram.py -train\" train \n',
              '\"python run_ngram.py -rescore\" rescore nbest\n',
              '\"python run_ngram.py -wer\" compute WER'
              )

    absdir = os.getcwd() + '/'
    bindir = absdir + '../../tools/srilm/'
    workdir = absdir + 'ngramlm/'
    wb.mkdir(workdir)


    datas = [absdir + i for i in data()]
    result_file = absdir + 'models_ppl.txt'  # the result file
    model = ngram.model(bindir, workdir)
    order_reg = [2, 3, 4, 5]

    for order in order_reg:
        write_model = workdir + '{}gram.lm'.format(order)
        print(write_model)

        if '-train' in sys.argv:
            if order_reg.index(order) == 0:
                model.prepare(datas[0], datas[1], datas[2])
            model.train(order, write_model, absdir + 'models_ppl.txt')
        if '-rescore' in sys.argv:
            model.rescore(write_model, order, datas[3], write_model[0:-3] + '.lmscore')
        if '-wer' in sys.argv:
            [nbest, templ] = datas[3:5]
            lmscore = wb.LoadScore(write_model[0:-3] + '.lmscore')
            acscore = wb.LoadScore(datas[5])

            [wer, lmscale, acscale] = wb.TuneWER(nbest, templ, lmscore, acscore, np.linspace(0.1,0.9,9))
            print('wer={} lmscale={} acscale={}'.format(wer, lmscale, acscale))
            fres = wb.FRes(result_file)
            fres.AddWER('KN{}'.format(order), wer)

            trans_txt = workdir + os.path.split(templ)[-1] + '.txt'
            wb.file_rmlabel(templ, trans_txt)
            PPL_temp = model.ppl(write_model, order, trans_txt)
            LL_temp = -wb.PPL2LL(PPL_temp, trans_txt)
            fres.Add('KN{}'.format(order), ['LL-wsj', 'PPL-wsj'], [LL_temp, PPL_temp])

if __name__ == '__main__':
    main()
