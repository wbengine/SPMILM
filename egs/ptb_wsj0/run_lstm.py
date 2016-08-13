 
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + '/../../tools/')
import lstm
import wb

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

  
def wer(workdir, datas):
  [read_nbest, read_trans, read_ac, read_lm] = datas[3:7]
  read_lm = workdir + read_nbest.split('/').pop() + '.lmscore'
  
  # compute the WER
  [wer, lmscale, acscale] = wb.TuneWER(read_nbest, read_trans, wb.LoadScore(read_lm), wb.LoadScore(read_ac), np.linspace(0.1, 0.9, 9) ) 
  print('wer={} lmscale={} acscale={}'.format(wer, lmscale, acscale))
  
  # compute the PPL on the test set
  read_vocab = workdir + 'vocab'
  read_model = workdir + 'small.lstm'
  write_trans = workdir + 'transcript.id'
  v = lstm.ReadVocab(read_vocab)
  lstm.GetNbest(read_trans, write_trans, v)
  os.system('th main.lua -vocab {} -read {} -test {}'.format(read_vocab, read_model, write_trans))
  
  
if __name__ == '__main__':  
  print(sys.argv)
  if len(sys.argv) == 1:
    print(' \"python run.py -train\" train LSTM\n \"python run.py -rescore\" rescore nbest\n \"python run.py -wer\" compute WER')
  
 
  absdir = os.getcwd() + '/'
  datas = [ absdir + i for i in data() ]
  
  workdir = absdir + '/lstmlm/'
  os.chdir('../../tools/lstm/')
  
  if not os.path.exists(workdir):
    os.mkdir(workdir)
  
  if '-train' in sys.argv:
    lstm.train(workdir, datas[0], datas[1], datas[2])
  if '-rescore' in sys.argv:
    lstm.rescore(workdir, datas[3], workdir + datas[3].split('/').pop()+'.lmscore')
  if '-wer' in sys.argv:
    wer(workdir, datas)