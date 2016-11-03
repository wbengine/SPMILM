import math
import os
import sys


# from itertools import zip_longest

# create the dir
def mkdir(path):
    a = path.split(os.sep)
    if path[0] == os.sep:
        s = path[0]
    else:
        s = ''
    for w in a:
        s += w + os.sep
        if not os.path.exists(s):
            os.mkdir(s)


# remove files
def remove(path):
    if os.path.exists(path):
        os.remove(path)


# make sure a folder
def folder(path):
    if path[-1] != os.sep:
        return path + os.sep
    return path


# get the name of current script
def script_name():
    argv0_list = sys.argv[0].split(os.sep)
    name = argv0_list[len(argv0_list) - 1]
    name = name[0:-3]
    return name


# count the sentence number and the word number of a txt files
def file_count(fname):
    f = open(fname)
    nLine = 0
    nWord = 0
    for line in f:
        nLine += 1
        nWord += len(line.split())
    f.close()
    return [nLine, nWord]


# rmove the frist column of each line
def file_rmlabel(fread, fout):
    with open(fread) as f1, open(fout, 'wt') as f2:
        for line in f1:
            f2.write(' '.join(line.split()[1:]) + '\n')


# get the word list in files
def getLext(fname):
    v = dict()
    f = open(fname)
    for line in f:
        words = line.split()
        for w in words:
            w = w.upper()  # note: to upper
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
def corpus_w2n(fin, fout, v, unk='<UNK>', id_offset=0):
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
                print('[wb.corpus_w2n]: cannot find the key = ' + w);
                exit()

            fo.write(''.join(['{} '.format(n + id_offset) for n in nums]))

        f.close()
        fo.close()


# corpus

# ppl to loglikelihood
# two usage: PPL2LL(ppl, nline, nword), PPL2LL(ppl, file)
def PPL2LL(ppl, obj1, obj2=0):
    nLine = obj1
    nWord = obj2
    if isinstance(obj1, str):
        [nLine, nWord] = file_count(obj1)
    return -math.log(ppl) * (nLine + nWord) / nLine


# LL to PPL
# two usage: LL2PPL(LL, nline, nword), LL2PPL(LL, file)
def LL2PPL(LL, obj1, obj2=0):
    nLine = obj1
    nWord = obj2
    if isinstance(obj1, str):
        [nLine, nWord] = file_count(obj1)
    return math.exp(-LL * nLine / (nLine + nWord))


# LL incrence bits to PPL decence precent
def LLInc2PPL(LLInc, obj1, obj2):
    nLine = obj1
    nWord = obj2
    if isinstance(obj1, str):
        [nLine, nWord] = file_count(obj1)
    return 1 - math.exp(-LLInc * nLine / (nLine + nWord))


# TxtScore: compare two word sequence (array), and return the error number
def TxtScore(target, base):
    res = {'word': 0, 'err': 0, 'none': 0, 'del': 0, 'ins': 0, 'rep': 0, 'target': [], 'base': []}

    target.insert(0, '<s>')
    target.append('</s>')
    base.insert(0, '<s>')
    base.append('</s>')
    nTargetLen = len(target)
    nBaseLen = len(base)

    if nTargetLen == 0 or nBaseLen == 0:
        return res

    aNext = [[0, 1], [1, 1], [1, 0]]
    aDistTable = [([['none', 10000, [-1, -1], '', '']] * nBaseLen) for i in range(nTargetLen)]
    aDistTable[0][0] = ['none', 0, [-1, -1], '', '']  # [error-type, note distance, best previous]

    for i in range(nTargetLen - 1):
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
                if dir == [0, 1]:
                    nextState = 'del'
                    nextScore += 1
                    nextTarget = '*' + ' ' * len(base[nextj])
                    nextBase = '*' + base[nextj]
                elif dir == [1, 0]:
                    nextState = 'ins'
                    nextScore += 1
                    nextTarget = '^' + target[nexti]
                    nextBase = '^' + ' ' * len(target[nexti])
                else:
                    nextTarget = target[nexti]
                    nextBase = base[nextj]
                    if target[nexti] != base[nextj]:
                        nextState = 'rep'
                        nextScore += 1
                        nextTarget = '~' + nextTarget
                        nextBase = '~' + nextBase

                if nextScore < aDistTable[nexti][nextj][1]:
                    aDistTable[nexti][nextj] = [nextState, nextScore, [i, j], nextTarget, nextBase]

    res['err'] = aDistTable[nTargetLen - 1][nBaseLen - 1][1]
    res['word'] = nBaseLen - 2
    i = nTargetLen - 1
    j = nBaseLen - 1
    while i >= 0 and j >= 0:
        res[aDistTable[i][j][0]] += 1
        res['target'].append(aDistTable[i][j][3])
        res['base'].append(aDistTable[i][j][4])
        [i, j] = aDistTable[i][j][2]
    res['target'].reverse()
    res['base'].reverse()

    return res


# calculate the WER given best file
def CmpWER(best, temp, log=''):
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
                fout.write(
                    '[{0}]	[nDist={1}]	[{1}/{2}]	[{3}/{4}]\n'.format(nLine, res['err'], res['word'], nTotalErr,
                                                                              nTotalWord))
                fout.write('Input: ' + ''.join([i + ' ' for i in res['target'][1:-1]]) + '\n')
                fout.write('Templ: ' + ''.join([i + ' ' for i in res['base'][1:-1]]) + '\n')

            nLine += 1
    return [nTotalErr, nTotalWord, 1.0 * nTotalErr / nTotalWord * 100]


# given the score get the 1-best result
def GetBest(nbest, score, best):
    f = open(nbest, 'rt')
    fout = open(best, 'wt')
    nline = 0
    bestscore = 0
    bestlabel = ''
    bestsent = ''
    for line in f:
        a = line.split()
        head = a[0]
        sent = ' '.join(a[1:])
        
        idx = head.rindex('-')
        label = head[0:idx]
        num = int(head[idx + 1:])
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


# Load the nbest/score label
def LoadLabel(fname):
    s = []
    f = open(fname, 'rt')
    for line in f:
        a = line.split()
        s.append(a[0])
    f.close()
    return s


# Write Score
def WriteScore(fname, s, label=[]):
    with open(fname, 'wt') as f:
        for i in range(len(s)):
            if len(label) == 0:
                f.write('line={}\t{}\n'.format(i, s[i]))
            else:
                f.write('{}\t{}\n'.format(label[i], s[i]))


# tune the lmscale and acscale to get the best WER
def TuneWER(nbest, temp, lmscore, acscore, lmscale, acscale=[1]):
    opt_wer = 100
    opt_lmscale = 0
    opt_acscale = 0
    if isinstance(lmscore, str):
        lmscore = LoadScore(lmscore)
    if isinstance(acscore, str):
        acscore = LoadScore(acscore)
    # tune the lmscale
    for acscale in acscale:
        for lmscale in lmscale:
            s = [0] * len(acscore)
            for i in range(len(s)):
                s[i] = acscale * acscore[i] + lmscale * lmscore[i]
            best_file = 'lm{}.ac{}.best'.format(lmscale, acscale)
            GetBest(nbest, s, best_file)

            [totale, totalw, wer] = CmpWER(best_file, temp)

            # print('acscale={}\tlmscale={}\twer={}\n'.format(acscale, lmscale, wer))
            if wer < opt_wer:
                opt_wer = wer
                opt_lmscale = lmscale
                opt_acscale = acscale

            # remove the best files
            os.remove(best_file)

    return [opt_wer, opt_lmscale, opt_acscale]


# Res file, such as
# model LL-train LL-valid LL-test PPL-train PPL-valid PPL-test
# Kn5  100 100 100 200 200 200
# rnn 100 100 100  200 200 200
class FRes:
    def __init__(self, fname, print_to_cmd=False):
        self.fname = fname  # the file name
        self.data = [] # recore all the data in files
        self.head = [] # recore all the label
        self.print_to_cmd = print_to_cmd
        self.new_add_name = ''  # record the current add name


    # load data from file
    def Read(self):
        self.data = []
        self.head = []
        if os.path.exists(self.fname):
            with open(self.fname, 'rt') as f:
                nline=0
                for line in f:
                    if nline == 0:
                        self.head = line.split()
                    else:
                        self.data.append(line.split())
                    nline+=1
        else:
            self.head.append('models')

        # return all the name in files
        names = []
        for a in self.data:
            names.append(a[0])
        return names


    # write data to file
    def Write(self):
        n = len(self.head)
        width = [len(i) for i in self.head]
        for a in self.data:
            for i in range(len(a)):
                width[i] = max(width[i], len(a[i]))
        with open(self.fname, 'wt') as f:
            for a in [self.head] + self.data:
                outputline = ''
                for i in range(len(a)):
                    outputline += '{0:{width}}'.format(a[i], width=width[i]+2)
                f.write(outputline + '\n')

                # print the new added line
                if self.print_to_cmd and a[0] == self.new_add_name:
                    print(outputline)

    # clean files
    def Clean(self):
        remove(self.fname)

    # remove default head
    def RMDefaultHead(self):
        self.head = ['models']


    # get the head
    def GetHead(self):
        self.Read()
        return self.head


    # get ['KN', '100', '111', 1213']
    def GetLine(self, name):
        self.Read()
        for a in self.data:
            if a[0] == name:
                return a
        print('[FRes] Cannot find {}'.format(name))
        return []


    # get ['KN', 100, 111, 1213]
    def Get(self, name):
        a = self.GetLine(name)
        res = [a[0]]
        for w in a[1:]:
            res.append(float(w))
        return res
    # add a line
    def AddLine(self, name):
        self.Read()
        for a in self.data:
            if a[0] == name:
                return a
        self.data.append([name])
        return self.data[-1]
    # add datas, such as Add('KN5', ['LL-train', 'LL-valid'], [100, 10] )
    def Add(self, name, head, value):
        a = self.AddLine(name)
        for w in head:
            if w not in self.head:
                self.head.append(w)
            i = self.head.index(w)
            
            if len(a) < len(self.head):
                a.extend(['0']* (len(self.head) - len(a)))
            v = value[ head.index(w) ]
            if isinstance(v, str):
                a[i] = v
            elif isinstance(v, float):
                a[i] = '{:.3f}'.format(float(v))
            else:
                a[i] = '{}'.format(v)
        self.new_add_name = name
        self.Write()

    def AddWER(self, name, wer):
        self.Add(name, ['WER'], [wer])
        
    def AddLL(self, name, LL, txt):
        PPL = [0] * len(LL)
        for i in range(len(LL)):
            [sents, words] = file_count(txt[i])
            PPL[i] = LL2PPL(-LL[i], sents, words)
        self.Add(name, [a+'-'+b for a in ['LL','PPL'] for b in ['train','valid','test']], LL+PPL)

    def AddPPL(self, name, PPL, txt):
        LL = [0] * len(PPL)
        for i in range(len(PPL)):
            [sents, words] = file_count(txt[i])
            LL[i] = -PPL2LL(PPL[i], sents, words)
        self.Add(name, [a+'-'+b for a in ['LL','PPL'] for b in ['train','valid','test']], LL+PPL)
