import theano
import theano.tensor as T
import numpy as np
import time
import wb
import os

def gd_momentum(cost, params, learning_rate, momentum=0.9):
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum*previous_step - learning_rate*T.grad(cost, param)
        #step = learning_rate*T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))

    return updates

def adam(cost, params, learning_rate):

    # parameters
    beta1 = 0.9
    beta2 = 0.999
    es = 10**-8

    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    prev_t = theano.shared(np.dtype(theano.config.floatX).type(0.))
    t = prev_t + 1
    updates.append((prev_t, t))

    for param in params:
        prev_m = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)
        prev_v = theano.shared(param.get_value()*0, broadcastable=param.broadcastable)

        g = T.grad(cost, param)

        cur_m = beta1*prev_m + (1-beta1)*g
        cur_v = beta2*prev_v + (1-beta2)*(g**2)

        m_tide = cur_m / (1-beta1**t)
        v_tied = cur_v / (1-beta2**t)
        step = learning_rate * m_tide / (T.sqrt(v_tied) + es)

        updates.append((prev_m, cur_m))
        updates.append((prev_v, cur_v))
        updates.append((param, param - step))

    return updates


def random_weights(shape):
    drange = np.sqrt(6. / (np.sum(shape)))
    return drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    #return np.zeros(shape)


def create_shared(value, name):
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def read_value(f, delimiter='='):
    line = f.next()
    return line.split('\n')[0].split(delimiter)[-1]


def write_array(f, a, name='array'):
    if a.ndim == 1:
        row = 1
        col = a.shape[0]
        m = a.reshape(row, col)
    elif a.ndim == 2:
        row = a.shape[0]
        col = a.shape[1]
        m = a
    else:
        print('[ERROR] cannot support a.ndim={}'.format(a.ndim))

    f.write('name={}\nndim={}\nrow={}\ncol={}\n'.format(name, a.ndim, row, col))
    np.savetxt(f, m, fmt='%.4f')


def read_array(f):
    name = read_value(f)
    dim = int(read_value(f))
    row = int(read_value(f))
    col = int(read_value(f))
    a = np.genfromtxt(f, dtype=theano.config.floatX, max_rows=row)
    if dim == 1:
        a = a.reshape((col,))
    elif dim == 2:
        a = a.reshape((row, col))
    return [name, a]



class SoftmaxLayer(object):
    """
    Simple neural network layer
    Without batches:
        Input : of shape (sequence_length, input_dim)
        Output: of shape (sequence_length, output_dim)
    With batches:
        Input : of shape (sequence_length, batch_size, input_dim)
        Output: of shape (sequence_length, batch_size, output_dim)
    """
    def __init__(self, input_dim, output_dim, with_batch=True, name='Layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_batch = with_batch
        self.name = name

        self.W = create_shared(random_weights((input_dim, output_dim)), name+'_W')
        self.b = create_shared(random_weights((output_dim, )), name+'_b')
        self.params = [self.W]

    def link(self, input):
        if self.with_batch:
            # as the input is a tensor3, to use T.nnet.softmax function, we need to reshape the output to a matrix
            y = T.dot(input, self.W) + self.b
            y = T.nnet.softmax(y.reshape((input.shape[0]*input.shape[1], self.output_dim)))
            self.output = y.reshape((input.shape[0], input.shape[1], self.output_dim))
        else:
            self.output = T.nnet.softmax(T.dot(input, self.W) + self.b)

        return self.output

class EmbeddingLayer(object):
    """
    Embedding layer
    Without batches:
        Input : of shape (sequence_length)
        Output: of shape (sequence_length, embed_dim)
    With batches:
        Input : of shape (sequence_length, batch_size)
        Output: of shape (sequence_length, batch_size, embed_dim)
    """
    def __init__(self, input_dim, embed_dim, with_batch=True, name='Embedding'):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.with_batch = with_batch
        self.name = name

        self.E = create_shared(random_weights((input_dim, embed_dim)), name + '_E')
        self.params = [self.E]

    def link(self, input):
        self.output = self.E[input, :]
        return self.output



class LSTMLayer(object):
    """
    LSTM with faster implementation (supposedly).
    Not as expressive as the previous one though, because it doesn't include the peepholes connections.
    Without batches:
        Input : of dimension (sequence_length, input_dim)
        Output: of dimension (sequence_length, hidden_dim)
    With batches:
        Input : of dimension (sequence_length, batch_size, input_dim)
        Output: of dimension (sequence_length, batch_size, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        self.W = create_shared(random_weights((input_dim, hidden_dim * 4)), name + '_W')
        self.U = create_shared(random_weights((hidden_dim, hidden_dim * 4)), name + '_U')
        self.b = create_shared(random_weights((hidden_dim * 4, )), name + '_b')

        self.c_0 = create_shared(np.zeros((hidden_dim,)), name + '__c_0')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        self.params = [self.W, self.U, self.b]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """
        def split(x, n, dim):
            if self.with_batch:
                return x[:, n*dim:(n+1)*dim]
            else:
                return x[n*dim:(n+1)*dim]

        def recurrence(x_t, c_tm1, h_tm1):
            p = x_t + T.dot(h_tm1, self.U)
            i = T.nnet.sigmoid(split(p, 0, self.hidden_dim))
            f = T.nnet.sigmoid(split(p, 1, self.hidden_dim))
            o = T.nnet.sigmoid(split(p, 2, self.hidden_dim))
            c = T.tanh(split(p, 3, self.hidden_dim))
            c = f * c_tm1 + i * c
            h = o * T.tanh(c)
            return c, h

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            # input     : tensor3 of shape (sequence_length, batch_size, input_dim)
            ones = T.ones((input.shape[1], 1))  # copy for each batch
            c_init = T.dot(ones, self.c_0.reshape((1, self.hidden_dim)))
            h_init = T.dot(ones, self.h_0.reshape((1, self.hidden_dim)))
            outputs_info = [c_init, h_init]
        else:
            outputs_info = [self.c_0, self.h_0]

        input = T.dot(input, self.W) + self.b

        [_, o_h], _ = theano.scan(
            fn=recurrence,
            sequences=input,
            outputs_info=outputs_info,
            n_steps=input.shape[0]
        )


        self.output = o_h

        return self.output

class LSTM:
    def __init__(self, vocab_size=1, hidden_dim=1, hidden_num=1, with_batch=True, name='[LSTM LM]', skip_bulid=False):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.with_batch = with_batch
        self.name = name

        self.layers = []
        self.layers.append(EmbeddingLayer(vocab_size, hidden_dim, with_batch=with_batch))
        for i in range(hidden_num):
            self.layers.append(LSTMLayer(hidden_dim, hidden_dim, with_batch=with_batch))
        self.layers.append(SoftmaxLayer(hidden_dim, vocab_size, with_batch=with_batch))
        self.params = []
        for i in self.layers:
            self.params += i.params

        if skip_bulid==False:
            self.bulid()

    def link(self, input):
        """
        without batches:
            input : of the shape (sequence_length) with dtype=int32
        with batches:
            input : of the shape (sequence_length, batch_size) with dtype=int32
        """
        output = input
        for layer in self.layers:
            output = layer.link(output)

        self.output = output

        return self.output

    def bulid(self):
        """
        Bulid function for the input sequence
        without batches:
            x,y : of the shape (sequence_length) with dtype=int32
        with batches:
            x.y : of the shape (sequence_length, batch_size) with dtype=int32
        """
        print('BULID...')
        if self.with_batch:
            data = T.imatrix('data')
            x = data[0:-1, :]
            y = data[1:, :]
        else:
            data = T.ivector('data')
            x = data[0:-1]
            y = data[1:]

        o = self.link(x)

        if self.with_batch:
            prep_o = o.reshape((o.shape[0]*o.shape[1], o.shape[2]))
            prep_y = y.reshape((y.shape[0]*y.shape[1],))
        else:
            prep_o = o
            prep_y = y

        cost = T.mean(T.nnet.categorical_crossentropy(prep_o, prep_y))
        nll =  T.sum(T.nnet.categorical_crossentropy(prep_o, prep_y))  # the negative log-likelihood
        # ppl = T.exp(nll / y.shape[0])

        # build function
        self.forward = theano.function([data], o)
        self.err = theano.function([data], cost)
        self.nll = theano.function([data], nll)
        self.grad = theano.function([data], T.grad(cost, self.params))

        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
        self.update = theano.function(inputs=[data, learning_rate],
                                     outputs=cost,
                                     updates=adam(cost, self.params, learning_rate)
                                     )

    def train(self, data_train, valid_list=None, test_list=None, sequence_len=20, max_epoch=10, learning_rate=0.001):
        """
        train LSTM on corpus
        :param data_train:
            without batch: of shape (sequence_total)
            with batch:    of shape (batch_size, sequence_total)
        :param data_valid: a list
        :param test_list:  a list
        :param sequence_len: the length of each sequence
        :param max_epoch:  maximum epoch number
        :param learning_rate: learning rate of adam algorithm
        :return:
        """
        if self.with_batch:
            sequence_total = data_train.shape[1]
            print('batch_size={}'.format(data_train.shape[0]))
        else:
            sequence_total = data_train.shape[0]
            print('batch_size=1')

        print('seq_total={}'.format(sequence_total))
        print('seq_len={}'.format(sequence_len))
        print('max_epoch={}'.format(max_epoch))

        print('Train...')
        time_total = 0.
        data_total = 0.
        epoch = 0.
        idx_cur = 0
        print_flag = 0
        eval_flag = 0
        while True:
            if self.with_batch:
                data_cur_iter = data_train[:, idx_cur: idx_cur+sequence_len].T
            else:
                data_cur_iter = data_train[idx_cur: idx_cur+sequence_len]

            idx_cur = idx_cur + sequence_len
            if idx_cur >= sequence_total:
                idx_cur = 0

            data_total += sequence_len
            epoch = 1.0 * data_total / sequence_total

            t_beg = time.time()
            err = self.update(data_cur_iter, learning_rate)
            time_total += time.time() - t_beg

            if epoch >= print_flag or epoch >= max_epoch:
                out_str = 'epoch={:.2f} time/epoch={:.2f}s iter_time={:.2f}s err_batch={:.2f} '.format(epoch, time_total/epoch, time_total, float(err))
                print(out_str)
                print_flag += 0.1

            if epoch >= eval_flag or epoch >= max_epoch:
                out_str = 'epoch={:.2f} iter_time={:.2f}s '.format(epoch, time_total)
                if valid_list is not None:
                    nll, ppl = self.eval(valid_list)
                    out_str += 'valid_nll={:.2f} valid_ppl={:.2f} '.format(nll, ppl)
                if test_list is not None:
                    nll, ppl = self.eval(test_list)
                    out_str += 'test_nll={:.2f} test_ppl={:.2f} '.format(nll, ppl)
                t_end = time.time()
                out_str += 'eval_time={:.2f}s'.format(t_end-t_beg)
                print(out_str)
                eval_flag += 1.0


            if epoch >= max_epoch:
                break

        print('Training Finished!')


    def eval(self, inputs):
        '''evaulate the NLL and PPL on the whole dataset'''
        logprobs = self.prob(inputs)
        seqlens = [len(x)-1 for x in inputs]

        nll = sum(logprobs)
        seq_len = sum(seqlens)

        ppl = np.exp(nll/seq_len)
        return [nll/seq_len, ppl]

    def prob(self, inputs):
        """ input a list of sequence, return the log-prob of each sequence """
        logprobs = []
        for x in inputs:
            assert np.max(x) >= 0
            assert np.max(x) < self.vocab_size
            x = np.array(x, dtype='int32')
            if self.with_batch:
                x = x.reshape(len(x),1)
            logprobs.append(self.nll(x))
        return logprobs

    def copy_params(self, net):
        for i in range(len(self.params)):
            self.params[i].set_value(net.params[i].get_value())

    def write(self, fname):
        if isinstance(fname, str):
            print('{} write to {}'.format(self.name, fname))
            f = open(fname, 'wt')
        else:
            f = fname

        f.write('name={}\n'.format(self.name))
        f.write('vocab_size={}\nhidden_dim={}\nhidden_num={}\nwith_batch={}\n'.format(
            self.vocab_size, self.hidden_dim, self.hidden_num, self.with_batch
        ))
        # write layers and parameters
        for layer in self.layers:
            for param in layer.params:
                a = param.get_value()
                write_array(f, a, param.name)

        if isinstance(fname, str):
            f.close()

    def read(self, fname):
        if isinstance(fname, str):
            print('{} read from {}'.format(self.name, fname))
            f = open(fname, 'rt')
        else:
            f = fname

        self.name = read_value(f)
        vocab_size = int(read_value(f))
        hidden_dim = int(read_value(f))
        hidden_num = int(read_value(f))
        with_batch = bool(read_value(f))
        if vocab_size != self.vocab_size \
                or hidden_dim != self.hidden_dim \
                or hidden_num != self.hidden_num:
            self.__init__(vocab_size, hidden_dim, hidden_num, with_batch, name=self.name)
        # read layers and parameters
        for layer in self.layers:
            for param in layer.params:
                p_name, a = read_array(f)
                param.set_value(a)
                param.name = p_name

        if isinstance(fname, str):
            f.close()


# exact the vocabulary form the corpus
def GetVocab(fname, v):
    f = open(fname, 'rt')
    for line in f:
        a = line.upper().split()
        for w in a:
            v.setdefault(w, 0)
            v[w] += 1
    f.close()
    return v


# set the vocab id
def SetVocab(v):
    n = 2
    for k in sorted(v.keys()):
        v[k] = n
        n += 1
    return v


# trans txt corpus to id corpus
def CorpusToID(fread, fwrite, v, unk='<UNK>'):
    if isinstance(v, str):
        v = ReadVocab(v)
    print('[w2id] ' + fread + ' -> ' + fwrite)
    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.upper().split()
        for w in a:
            if w in v:
                f2.write('{} '.format(v[w]))
            else:
                f2.write('{} '.format(v[unk]))
        f2.write('\n')
    f1.close()
    f2.close()


# trans id to txt
def CorpusToW(fread, fwrite, v):
    if isinstance(v, str):
        v = ReadVocab(v)
    print('[id2w] ' + fread + ' -> ' + fwrite)
    v1 = [''] * (len(v) + 2)
    for key in v.keys():
        v1[v[key]] = key

    f1 = open(fread, 'rt')
    f2 = open(fwrite, 'wt')
    for line in f1:
        a = line.split()
        for w in a:
            f2.write('{} '.format(v1[int(w)]))
        f2.write('\n')
    f1.close()
    f2.close()


# write vocabulary
def WriteVocab(fname, v):
    f = open(fname, 'wt')
    vlist = sorted(v.items(), key=lambda d: d[1])
    f.write('<s>\n</s>\n')
    for w, wid in vlist:
        f.write('{}\t{}\n'.format(wid, w))
    f.close()


# read vocabulary
def ReadVocab(fname):
    v = dict()
    f = open(fname, 'rt')
    f.readline()
    f.readline()
    for line in f:
        a = line.split()
        v[a[1].upper()] = int(a[0])
    f.close()
    return v


# trans nbest list to id files
def GetNbest(ifile, ofile, v, unk='<UNK>'):
    if isinstance(v, str):
        v = ReadVocab(v)
    print('[nbest] ' + ifile + ' -> ' + ofile)
    fin = open(ifile, 'rt')
    fout = open(ofile, 'wt')
    for line in fin:
        a = line.upper().split()
        for w in a[1:]:
            nid = 0
            if w in v:
                nid = v[w]
            elif unk in v:
                nid = v[unk]
            else:
                print('[error] on word in vocabulary ' + w)
            fout.write('{} '.format(nid))
        fout.write('\n')
    fin.close()
    fout.close()

def load_data(name):

    w = []
    with open(name) as f:
        for line in f:
            a = [int(i) for i in line.split()]
            a.insert(0, 0)  # add the beg-token
            a.append(1)     # append the end-token
            w.append(a)
    print('load sequence = {} form {}'.format(len(w), name))

    return w


def trans_data(w, batch_size):

    # shuffle
    np.random.shuffle(w)
    x = []
    for a in w:
        x += a

    # split
    n = len(x)
    seq_total_len = int(np.ceil(1.0 * n / batch_size))
    n_new = seq_total_len * batch_size
    data = np.zeros(n_new, dtype='int32')
    data[0:n] = x
    data = data.reshape(batch_size, seq_total_len)
    return data


class model:
    def __init__(self, workdir):
        self.workdir = wb.folder(workdir)
        self.net = None
        wb.mkdir(workdir)

    def prepare(self, train, valid, test, nbest=''):
        v = dict()
        GetVocab(train, v)
        GetVocab(valid, v)
        GetVocab(test, v)
        SetVocab(v)
        WriteVocab(self.workdir + 'vocab', v)

        CorpusToID(train, self.workdir + 'train.no', v)
        CorpusToID(valid, self.workdir + 'valid.no', v)
        CorpusToID(test, self.workdir + 'test.no', v)

        if os.path.exists(nbest):
            GetNbest(nbest, self.workdir + 'nbest.no', v)

    def train(self, hidden_dim, hidden_num, write_model=None, batch_size=20, sequence_len=20, max_epoch=10, learning_rate=0.001):
        """
        traing a lstm using theano code.
        :param hidden_dim: hidden dimension
        :param hidden_num: hidden layer number
        :param batch_size: batch_size used in batch training
        :param sequence_len: sequence_len used in training
        :param write_model: write to txt files
        :return:
        """
        v = ReadVocab(self.workdir + 'vocab')
        vocab_size = len(v) + 2 # add the <s> and </s>
        print('vocab_size={}'.format(vocab_size))

        self.net = LSTM(vocab_size, hidden_dim, hidden_num, with_batch=True)
        self.net.train(data_train=trans_data(load_data(self.workdir + 'train.no'), batch_size),
                       valid_list=load_data(self.workdir + 'valid.no'),
                       test_list =load_data(self.workdir + 'test.no'),
                       sequence_len=sequence_len,
                       max_epoch=max_epoch,
                       learning_rate=learning_rate
                       )
        nll, ppl = self.net.eval(load_data(self.workdir + 'test.no'))
        print('nll on test = {}'.format(nll))
        print('ppl on test = {}'.format(ppl))

        if write_model is not None:
            self.net.write(write_model)

    def ppl(self, test, read_model=None):
        [a, b] = os.path.split(test)
        write_test = self.workdir + b + '.pplid'

        CorpusToID(test, write_test, self.workdir + 'vocab')

        if self.net is None:
            self.net = LSTM(skip_bulid=True)
        if read_model is not None:
            self.net.read(read_model)
        _, ppl = self.net.eval(load_data(write_test))
        return ppl

    def rescore(self, nbest, rescore, read_model=None):
        print('resocre...')
        write_nbest_no = self.workdir + 'nbest.no'

        GetNbest(nbest, write_nbest_no, self.workdir + 'vocab')

        if self.net is None:
            self.net = LSTM(skip_bulid=True)
        if read_model is not None:
            self.net.read(read_model)
        logprobs = self.net.prob(load_data(write_nbest_no))

        cur = 0
        with open(nbest, 'rt') as fin, open(rescore, 'wt') as fout:
            for line in fin:
                label = line.split()[0]
                fout.write('{}\t{}\n'.format(label, logprobs[cur]))
                cur += 1

        assert(cur == len(logprobs))



def test():
    vocab_size = 2
    hidden_dim = 3
    hidden_num = 1

    lstm_net = LSTM(vocab_size, hidden_dim, hidden_num, with_batch=False)

    np.random.seed(0)
    mE = np.random.uniform(-0.1, 0.1, (vocab_size, hidden_dim))
    mT = np.random.uniform(-0.1, 0.1, (2*hidden_dim, 4*hidden_dim))
    mV = np.random.uniform(-0.1, 0.1, (hidden_dim, vocab_size))
    mTb = np.random.uniform(-0.1, 0.1, 4*hidden_dim)
    mVb = np.random.uniform(-0.1, 0.1, vocab_size)

    # lstm_net.layers[0].E.set_value(mE.astype(theano.config.floatX))
    # lstm_net.layers[1].W.set_value(mT[0:hidden_dim,:].astype(theano.config.floatX))
    # lstm_net.layers[1].U.set_value(mT[hidden_dim:,:].astype(theano.config.floatX))
    # lstm_net.layers[1].b.set_value(mTb.astype(theano.config.floatX))
    # lstm_net.layers[2].W.set_value(mV.astype(theano.config.floatX))
    # lstm_net.layers[2].b.set_value(mVb.astype(theano.config.floatX))

    x = np.array([1,0,1,0,0], dtype='int32')
    print(lstm_net.forward(x[0:-1]))
    print(lstm_net.err(x[0:-1], x[1:]))
    print(lstm_net.nll(x[0:-1], x[1:]))

    for t in range(100000):
        err = lstm_net.update(x[0:-1], x[1:], 0.1)
        print('t={}  err={}'.format(t, err))



def main_without_batch():
    np.random.seed(0)
    datadir = 'D:\\wangbin\\work\\HRFLM\\egs\\Word\\ngramlm\\'
    train_w = load_data(datadir + 'train.no')
    valid_w = load_data(datadir + 'valid.no')
    test_w = load_data(datadir + 'test.no')

    hidden_dim = 160
    hidden_layer = 1
    minibatch = 20
    sequence_len = 20
    max_epoch = 10
    write_txt = 'lstm-LM.txt'
    data_train = trans_data(train_w, 1).flatten()
    vocab_size = int(np.max(data_train)+1)

    print('vocab_size={}'.format(vocab_size))

    print('Build...')
    lstm_net = LSTM(vocab_size, hidden_dim, hidden_layer, with_batch=False)

    lstm_net.train(data_train, valid_w, sequence_len=sequence_len, max_epoch=max_epoch)

def main_batch():
    """
    A example to apply batch train
    :return: None
    """
    np.random.seed(0)
    datadir = 'ngramlm/'
    workdir = 'lstmlm/'
    train_w = load_data(datadir + 'train.no')
    valid_w = load_data(datadir + 'valid.no')
    test_w = load_data(datadir + 'test.no')
    nbest_w = load_data(datadir + 'nbest.no')

    hidden_dim = 16
    hidden_layer = 1
    minibatch = 20
    sequence_len = 20
    max_epoch = 1
    write_txt = workdir + 'lstm-LM.txt'
    data_train = trans_data(train_w, minibatch)
    vocab_size = int(np.max(data_train)+1)

    print('vocab_size={}'.format(vocab_size))

    print('Build...')
    lstm_net = LSTM(vocab_size, hidden_dim, hidden_layer, with_batch=True)
    #lstm_net.train(data_train, valid_w, sequence_len=sequence_len, max_epoch=max_epoch)

    nll, ppl = lstm_net.eval(test_w)
    print('nll on test = {}\nppl on test = {}'.format(nll, ppl))
    lstm_net.write(write_txt)

    # rescore


if __name__ == '__main__':
    # test()
    # main_without_batch()
    main_batch()

