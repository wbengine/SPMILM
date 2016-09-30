import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import ngram


# revise this function to config the dataset used to train different model
def data():
    root = './data/'
    train = root + 'train'
    valid = root + 'valid'
    test = root + 'valid'
    return data_verfy([train, valid, test])



def data_verfy(paths):
    for w in paths:
        if not os.path.isfile(w):
            print('[ERROR] no such file: ' + w)
    return paths


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
            model.train(order, write_model, result_file)

if __name__ == '__main__':
    main()
