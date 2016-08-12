# LM toolkits summarized by Bin Wang, SPMI Lab, Tsinghua University, Beijing, China
 
This toolkits includes the state-of-the-art LM methods and scripts for LM training, testing and nbest list rescoring.
This toolkits can be directly used to the evaulation different LMs by rescoring nbest list and to compare the speech recognition WER.
 
The LM in this toolkits includes:
1. LSTM: the Long Short-Term Memory nerual network language models. 
  * The source codes are released by Wojciech Zaremba at https://github.com/wojzaremba/lstm. We extand the code to make it easy to nbest rescoring. We evaulate LSTM in several datasets including PTM and WSJ0. 
  * Please install Torch http://torch.ch/ before using this LSTM toolkits.

The package are originazed as follows:
-tools
-tools/lstm
-egs
-egs/ptb_wsj0
-egs/chime4

'tools' includes all the toolkits include LSTM. 
'egs' includes all the experiments data, including scripts (written by python) and data (lm training corpus and nbest list used to rescore)
'egs/ptb_wsj0': Training LMs on PTB dataset and evaulate the WER on WSJ0 nbest list. See: Trans-dimensional random fields for language modeling, ACL2015
'egs/chime4': Run chime4 experiments. 