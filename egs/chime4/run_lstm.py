 
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd() + '/../../tools/')
import lstm
import wb
import wer

def rescore_all(workdir, nbestdir):
  for tsk in [ 'nbestlist_{}_{}'.format(a,b) for a in ['dt05', 'et05'] for b in ['real', 'simu'] ]:
    print('process ' + tsk)
    nbest_txt = nbestdir + tsk + '/words_text'
    outdir = workdir + tsk + '/'
    wb.mkdir(outdir)
    
    write_lmscore = outdir + 'lmwt.lstm'
    lstm.rescore(workdir, nbest_txt, write_lmscore)
  
  
if __name__ == '__main__':  
  print(sys.argv)
  if len(sys.argv) == 1:
    print(' \"python run.py -train\" train LSTM\n \"python run.py -rescore\" rescore nbest\n \"python run.py -wer\" compute WER')
  
 
  absdir = os.getcwd() + '/'
  train = absdir + 'data/train'
  valid = absdir + 'data/valid'
  nbestdir = absdir + 'data/NBEST_HEQ/'
  workdir = absdir + 'lstmlm/'
  os.chdir('../../tools/lstm/')
  
  if not os.path.exists(workdir):
    os.mkdir(workdir)
  
  if '-train' in sys.argv:
    lstm.train(workdir, train, valid, valid)
  if '-test' in sys.argv:
    lstm.ppl(workdir, train)
    lstm.ppl(workdir, valid)
  if '-rescore' in sys.argv:
    rescore_all(workdir, nbestdir)
  if '-wer' in sys.argv:
    lmpaths = {'KN5': nbestdir + '<tsk>/lmwt.lmonly', 
	       'RNN': nbestdir +'<tsk>/lmwt.rnn',
	       'LSTM': workdir + '<tsk>/lmwt.lstm'}
    lmtypes = ['KN5', 'RNN', 'RNN+KN5']
    wer_workdir = absdir + 'wer/'
    #wer.wer_all(wer_workdir, nbestdir, lmpaths, lmtypes)
    config = wer.wer_tune(wer_workdir)
    wer.wer_print(wer_workdir, config)