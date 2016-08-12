import math
import os
import sys
#from itertools import zip_longest

# create the dir
def mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)
    
# get the name of current script
def script_name():
  argv0_list = sys.argv[0].split('\\')
  name = argv0_list[len(argv0_list)-1]
  name = name[0:-3]
  return name
	

#count the sentence number and the word number of a txt files
def file_count(fname):
  f = open(fname)
  nLine = 0
  nWord = 0
  for line in f:
    nLine += 1
    nWord += len(line.split())
  f.close()
  return [nLine, nWord]
	
#get the word list in files
def getLext(fname):
  v = dict()
  f = open(fname)
  for line in f:
    words = line.split()
    for w in words:
      w = w.upper() #note: to upper
      n = v.setdefault(w, 0)
      v[w] = n + 1
  f.close()

  # resorted
  n = 0
  for k in sorted(v.keys()):
    v[k] = n
    n += 1
  return v

# corpus word to number
# the id of a word w  = v[w] + id_offset (if w in v) or v[unk]+ id_offset (if w not in v)
def corpus_w2n(fin, fout, v, unk = '<UNK>', id_offset = 0):
  f = open(fin)
  fo = open(fout, 'wt')
  for line in f:
    words = line.split()
    nums = []
    for w in words:
      w = w.upper()
      if w in v:
	nums.append(v[w])
      elif unk in v:
	nums.append(v[unk])
      else:
	print('[wb.corpus_w2n]: cannot find the key = '+w);
	exit()
	
      fo.write( ''.join(['{} '.format(n+id_offset) for n in nums]) )
      
    f.close()
    fo.close()
# corpus
	
#ppl to loglikelihood
def PPL2LL(ppl, nLine, nWord):
  return -math.log(ppl) * (nLine+nWord) / nLine
#LL to PPL
def LL2PPL(LL, nLine, nWord):
  return math.exp(-LL * nLine / (nLine+nWord))
	
#LL incrence bits to PPL decence precent
def LLInc2PPL(LLInc, nLine, nWord):
	return 1-math.exp(-LLInc * nLine / (nLine+nWord))
	
#TxtScore: compare two word sequence (array), and return the error number
def TxtScore(target, base):
	res = {'word':0, 'err':0, 'none':0, 'del':0, 'ins':0, 'rep':0, 'target':[], 'base':[]}
	
	target.insert(0, '<s>')
	target.append('</s>')
	base.insert(0, '<s>')
	base.append('</s>')
	nTargetLen = len(target)
	nBaseLen = len(base)
	
	if nTargetLen==0 or nBaseLen==0:
		return res
	
	aNext = [[0,1], [1,1], [1,0]]
	aDistTable = [ ([['none', 10000, [-1,-1], '', '']]*nBaseLen) for i in range(nTargetLen) ]
	aDistTable[0][0] = ['none', 0, [-1,-1], '', ''] # [error-type, note distance, best previous]
	
	for i in range(nTargetLen-1):
		for j in range(nBaseLen):
			for dir in aNext:
				nexti = i + dir[0]
				nextj = j + dir[1]
				if nexti >= nTargetLen or nextj >= nBaseLen:
					continue

				nextScore = aDistTable[i][j][1]
				nextState = 'none'
				nextTarget = ''
				nextBase = ''
				if dir == [0,1]:
					nextState = 'del'
					nextScore += 1
					nextTarget = '*' + ' '*len(base[nextj])
					nextBase = '*' + base[nextj]
				elif dir == [1,0]:
					nextState = 'ins'
					nextScore += 1
					nextTarget = '^' + target[nexti]
					nextBase = '^' + ' '*len(target[nexti])
				else:
					nextTarget = target[nexti]
					nextBase = base[nextj]
					if target[nexti] != base[nextj]:
						nextState = 'rep'
						nextScore += 1
						nextTarget = '~' + nextTarget
						nextBase = '~' + nextBase
				
				if nextScore < aDistTable[nexti][nextj][1]:
					aDistTable[nexti][nextj] = [nextState, nextScore, [i,j], nextTarget, nextBase]
	
	
	res['err'] =  aDistTable[nTargetLen-1][nBaseLen-1][1]
	res['word'] = nBaseLen - 2
	i = nTargetLen-1
	j = nBaseLen-1
	while i>=0 and j>=0:
		res[aDistTable[i][j][0]] += 1
		res['target'].append(aDistTable[i][j][3])
		res['base'].append(aDistTable[i][j][4])
		[i,j] = aDistTable[i][j][2]
	res['target'].reverse()
	res['base'].reverse()
	
	return res

#calculate the WER given best file
def CmpWER(best, temp, log = ''):
	nLine = 0
	nTotalWord = 0;
	nTotalErr = 0;
	fout = 0
	
	if log != '':
		fout = open(log, 'wt')
		
	with open(best) as f1, open(temp) as f2:
		for line1, line2 in zip(f1, f2):
			res = TxtScore(line1.split()[1:], line2.split()[1:])
			nTotalErr += res['err']
			nTotalWord += res['word']
			
			if log != '':
				fout.write('[{0}]	[nDist={1}]	[{1}/{2}]	[{3}/{4}]\n'.format(nLine, res['err'], res['word'], nTotalErr, nTotalWord ))
				fout.write('Input: ' + ''.join([i+' ' for i in res['target'][1:-1]]) + '\n')
				fout.write('Templ: ' + ''.join([i+' ' for i in res['base'][1:-1]]) + '\n')
				
			nLine+=1
	return [nTotalErr, nTotalWord, 1.0*nTotalErr/nTotalWord * 100]

#given the score get the 1-best result
def GetBest(nbest, score, best):
	f = open(nbest, 'rt')
	fout = open(best, 'wt')
	nline = 0
	bestscore = 0
	bestlabel = ''
	bestsent = ''
	for line in f:
		line = line[0:-1] # remove the '\n'
		[head, sep, sent] = line.partition(' ')

		idx = head.rindex('-')
		label = head[0:idx]
		num = int(head[idx+1:])
		if num == 1:
			if nline > 0:
				fout.write('{} {}\n'.format(bestlabel, bestsent))
			bestscore = score[nline]
			bestlabel = label
			bestsent = sent
		else:
			if score[nline] < bestscore:
				bestscore = score[nline]
				bestsent = sent
		nline += 1
	fout.write('{} {}\n'.format(bestlabel, bestsent))
	f.close()
	fout.close()	
	
# load the score file
def LoadScore(fname):
	s = []
	f = open(fname, 'rt')
	for line in f:
		a = line.split()
		s.append(float(a[1]))
	f.close()
	return s

#tune the lmscale and acscale to get the best WER
def TuneWER(nbest, temp, lmscore, acscore, lmscale, acscale = [1]):
	opt_wer = 100
	opt_lmscale = 0
	opt_acscale = 0
	#tune the lmscale
	for acscale in acscale:
		for lmscale in lmscale:
			s = [0] * len(acscore)
			for i in range(len(s)):
				s[i] = acscale*acscore[i] +  lmscale*lmscore[i]
			best_file = 'temp{}.best'.format(lmscale)
			GetBest(nbest, s, best_file)
			
			[totale, totalw, wer] = CmpWER(best_file, temp)
			
			#print('acscale={}\tlmscale={}\twer={}\n'.format(acscale, lmscale, wer))
			if wer < opt_wer:
				opt_wer = wer
				opt_lmscale = lmscale
				opt_acscale = acscale
				
			
			#remove the best files
			os.remove(best_file)

	return [opt_wer, opt_lmscale, opt_acscale]
	













	