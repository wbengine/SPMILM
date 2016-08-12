 
import os
import sys
import numpy as np

# exact the vocabulary form the corpus
def GetVocab(fname, v):
  f = open(fname, 'rt')
  for line in f:
    a = line.upper().split()
    a.append('<eos>'.upper()) # sentence tail symbol
    for w in a:
      v.setdefault(w, 0)
      v[w] += 1
  f.close()
  return v

# trans txt corpus to id corpus
def CorpusToID(fread, fwrite, v):
  print('[txt] ' + fread + ' -> ' + fwrite)
  f1 = open(fread, 'rt')
  f2 = open(fwrite, 'wt')
  for line in f1:
    a = line.upper().split()
    a.append('<eos>'.upper()) # sentence tail symbol
    for w in a:
      f2.write('{} '.format(v[w]))
    f2.write('\n')
  f1.close()
  f2.close()

# write vocabulary
def WriteVocab(fname, v):
  f = open(fname, 'wt')
  for k in sorted(v.keys()):
    f.write('{}\t{}\n'.format(k,v[k]))
  f.close()
 
# read vocabulary
def ReadVocab(fname):
  v = dict()
  f = open(fname, 'rt')
  for line in f:
    a = line.split()
    v[a[0].upper()] = int(a[1])
  f.close()
  return v

# trans nbest list to id files
def GetNbest(ifile, ofile, v, unk = '<UNK>'):
  print('[nbest] ' + ifile + ' -> ' + ofile)
  fin = open(ifile, 'rt')
  fout = open(ofile, 'wt')
  for line in fin:
    a = line.upper().split()[1:]
    a.append('<eos>'.upper())
    for w in a:
      nid = v[unk]
      if w in v:
	nid = v[w]
      fout.write('{} '.format(nid))
    fout.write('\n')
  fin.close()
  fout.close()
  
# train lstm
# revise the lstm config in file: ./tools/lstm/main.lua
def train(workdir, train, valid, test):
  write_vocab = workdir + 'vocab'
  write_model = workdir + 'small.lstm'
  write_train = workdir + 'train.id'
  write_valid = workdir + 'valid.id'
  write_test = workdir + 'test.id'
  
  # get the vocab
  v = dict()
  GetVocab(train, v)
  GetVocab(valid, v)
  GetVocab(test, v)
  
  # set the word-id
  n = 0
  for k in sorted(v.keys()):
    v[k] = n + 1
    n += 1
    
  # write vocab to file
  WriteVocab(write_vocab, v)
  
  # corpus to id
  CorpusToID(train, write_train, v)
  CorpusToID(valid, write_valid, v)
  CorpusToID(test, write_test, v)
  
  # train lstm
  cmd = 'th main.lua -vocab {} -train {} -valid {} -test {} -write {}'.format(write_vocab, write_train, write_valid, write_test, write_model)
  print(cmd)
  os.system(cmd)
  return write_model

# calculate the PPL of given files
def ppl(workdir, txt):
  read_vocab = workdir + 'vocab'
  read_model = workdir + 'small.lstm'
  read_txt = txt
  
  v = ReadVocab(read_vocab)
  write_txt = workdir + read_txt.split('/').pop() + '.pplid'
  CorpusToID(read_txt, write_txt, v)
  cmd = 'th main.lua -vocab {} -read {} -test {}'.format(read_vocab, read_model, write_txt)
  os.system(cmd)
  
  
# rescore the nbest list and get the lmscore, input the txt nbest file
def rescore(workdir, nbest, lmscore):
  read_vocab = workdir + 'vocab'
  read_model = workdir + 'small.lstm'
  read_nbest_txt = nbest
  write_nbest_id = workdir + read_nbest_txt.split('/').pop() + '.id'
  write_lmscore = lmscore
    
  v = ReadVocab(read_vocab)
  GetNbest(read_nbest_txt, write_nbest_id, v)
  cmd = 'th main.lua -vocab {} -read {} -nbest {} -score {} '.format(read_vocab, read_model, write_nbest_id, write_lmscore)
  os.system(cmd)
  
  return write_lmscore