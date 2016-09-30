import os
import sys

sys.path.insert(0, os.getcwd() + '/../../tools/')
import wb


def GetVocab(ftxt, v):
    # print('  [GetVocab] from ' + ftxt)
    f = open(ftxt, 'rt')
    for line in f:
        a = line.split()
        for w in a:
            v.setdefault(w, 0)
            v[w] += 1
    f.close()


def CutVocab(v, maxnum):
    # print('  [CutVocab] maxnum={}'.format(maxnum))
    vlist = v.items()
    vlist = sorted(vlist, key=lambda d: d[0], reverse=False)
    vlist = sorted(vlist, key=lambda d: d[1], reverse=True)
    for i in range(maxnum, len(vlist)):
        key = vlist[i][0]
        count = vlist[i][1]
        del v[key]
    v['<unk>'] = 0  # add unk
    return vlist


def WriteVocab(fwrite, v):
    # print('  [write] -> ' + fwrite)
    f = open(fwrite, 'wt')
    nid = 0
    for key in sorted(v.keys()):
        f.write('{}\t{}\n'.format(nid, key))
        nid += 1
    f.close()


def CutTxt(fread, fwrite, v, unk='<unk>'):
    # print('  [cut-txt] {} -> {}'.format(fread, fwrite))
    f1 = open(fread)
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.split()
        for i in range(len(a)):
            if a[i] not in v:
                a[i] = unk
        f2.write(' '.join(a) + '\n')
    f1.close()
    f2.close()


def GetTrainTxt(datadir, fwrite, num):
    # print('  [merge] -> {}'.format(fwrite))
    a = os.listdir(datadir)
    files = sorted([datadir + x for x in a if os.path.isfile(datadir + x)])

    cmd = 'cat '
    for i in range(min(num, len(files))):
        cmd += files[i] + ' '
    cmd += ' > ' + fwrite
    print(cmd)
    os.system(cmd)


# main
def main():
    dataroot = 'data/1-billion/'
    traindir = dataroot + 'training-monolingual.tokenized.shuffled/'
    valid_txt = dataroot + 'heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050'
    test_txt = dataroot + 'heldout-monolingual.tokenized.shuffled/news.en.heldout-00001-of-00050'

    for tsize in [1, 2, 4]:
        print('tsk = {}'.format(tsize))
        tskdir = '{}/'.format(tsize)
        wb.mkdir(tskdir)
        wb.mkdir(tskdir + 'data')

        write_train_all = tskdir + 'data/train.txt.all'
        write_train = tskdir + 'data/train.txt'
        write_valid = tskdir + 'data/valid.txt'
        write_test = tskdir + 'data/test.txt'
        write_count = tskdir + 'data/train.unigram'

        GetTrainTxt(traindir, write_train_all, tsize)

        v = dict()
        GetVocab(write_train_all, v)
        CutVocab(v, 20000 - 1)  # leave a space of <unk>
        WriteVocab(write_count, v)

        CutTxt(write_train_all, write_train, v)
        CutTxt(valid_txt, write_valid, v)
        CutTxt(test_txt, write_test, v)


if __name__ == '__main__':
    main()
