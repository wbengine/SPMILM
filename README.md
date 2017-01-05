# LM toolkits summarized by Bin Wang, SPMI Lab, Tsinghua University, Beijing, China
 
This toolkits includes the source code of the state-of-the-art langauge models, including ngram LMs, RNN LMs, LSTM LM and TRF LM. 
This toolkits can be directly used to the evaulation different LMs by rescoring nbest list and to compare the speech recognition WER.
Install:
```
./install.sh
```
This will install the SRILM, RNNLM and TRF.
To use LSTM LM, please install Torch. See http://torch.ch/ before using this LSTM toolkits.
 

The LM in this toolkits includes:

1. SRILM: a open toolkits to training ngram language models
2. RNNLM: the recurrent neural network language models
  * The source codes are released by Tomas Mikolov at http://www.fit.vutbr.cz/~imikolov/rnnlm/
3. LSTM: the Long Short-Term Memory nerual network language models. 
  * The source codes are released by Wojciech Zaremba at https://github.com/wojzaremba/lstm. We extand the code to make it easy to nbest rescoring. We evaulate LSTM in several datasets including PTM and WSJ0. 
  * Please install Torch http://torch.ch/ before using this LSTM toolkits.
4. TRF: the trans-dimensional random fields models
  * TRF LMs are a new kind of whole-sentence LMs which have the potential to integrate a richer set of features. 
  * Several experiments have shown that interpplating TRF with LSTM achieves the new state-of-the-art WER on speech recognition tasks.
  * See ''Trans-dimensional random fields for language modeling, ACL2015'' for TRF details
  * See ''Model Interpolation with Trans-dimensional Random Field Language Models for Speech Recognition, arXiv:1603.09170'' for more experiment results.


The package are originazed as follows:
```
-tools
-tools/srilm
-tools/rnn
-tools/lstm
-tools/trf
-egs
-egs/ptb_wsj0
-egs/chime4
-egs/1-billion
```

- 'tools' includes all the source of all toolkits
- 'egs' includes all the experiments data, including scripts (written by python) and data (lm training corpus and nbest list used to rescore)
- 'egs/ptb_wsj0': Training LMs on PTB dataset and evaulate the WER on WSJ0 nbest list. See: Trans-dimensional random fields for language modeling, ACL2015
- 'egs/chime4': Run chime4 experiments. 
- 'egs/1-billion': the experiments on Google 1-billion dataset
