import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb
import rnn


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


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 1:
        print(' \"python run_rnn.py -train\"   train rnn\n '
              ' \"python run_rnn.py -test\"    get the ppl\n '
              ' \"python run_rnn.py -rescore\" rescore nbest\n'
              ' \"python run_rnn.py -wer\"         compute WER'
              )
    absdir = os.getcwd() + '/'
    fres = wb.FRes('result.txt')

    for tsize in [1, 2, 4]:
        bindir = '../../tools/rnn/rnnlm-0.3e/'
        tskdir = absdir + '{}/'.format(tsize)
        workdir = tskdir + 'rnnlm/'
        model = rnn.model(bindir, workdir)

        hidden = 250
        cnum = 1
        bptt = 4
        write_model = workdir + 'h{}_c{}_bptt{}.rnn'.format(hidden, cnum, bptt)
        write_name = '{}:RNN:'.format(tsize) + os.path.split(write_model)[1][0:-4]

        if '-train' in sys.argv or '-all' in sys.argv:
            if not os.path.exists(write_model):
                config = ' -rnnlm {} '.format(write_model)
                config += ' -hidden {} -class {} -bptt {} '.format(hidden, cnum, bptt)
                config += ' -rand-seed 1'
                # config += ' -debug 2 -bptt-block 10 -direct-order 3 -direct 2 -binary'

                wb.remove(write_model)
                wb.remove(write_model + '.output.txt')
                model.prepare(data(tskdir)[0], data(tskdir)[1], data(tskdir)[2], data(tskdir)[3])
                model.train(config)
            else:
                print('exist model: ' + write_model)

        if '-test' in sys.argv or '-all' in sys.argv:
            PPL = [0]*3
            PPL[0] = model.ppl(write_model, data(tskdir)[0])
            PPL[1] = model.ppl(write_model, data(tskdir)[1])
            PPL[2] = model.ppl(write_model, data(tskdir)[2])
            fres.AddPPL(write_name, PPL, data(tskdir)[0:3])

        if '-rescore' in sys.argv or '-all' in sys.argv:
            write_lmscore = write_model[0:-4] + '.lmscore'
            model.rescore(write_model, data(tskdir)[3], write_lmscore)

        if '-wer' in sys.argv or '-all' in sys.argv:
            [read_nbest, read_templ, read_acscore, read_lmscore] = data(tskdir)[3:7]
            read_lmscore = write_model[0:-4] + '.lmscore'

            [wer, lmscale, acscale] = wb.TuneWER(read_nbest, read_templ,
                                                 read_lmscore, read_acscore, np.linspace(0.1, 0.9, 9))
            print('wer={:.4f} lmscale={:.2f} acscale={:.2f}'.format(wer, lmscale, acscale))
            fres.AddWER(write_name, wer)

            write_templ_rm = workdir + os.path.split(read_templ)[1] + '.rmlabel'
            rnn.Nbest_rmlabel(read_templ, write_templ_rm)
            PPL_templ = model.ppl(write_model, write_templ_rm)
            LL_templ = -wb.PPL2LL(PPL_templ, write_templ_rm)
            fres.Add(write_name, ['LL-wsj', 'PPL-wsj'], [LL_templ, PPL_templ])
