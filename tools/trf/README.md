Trans-dimensional random field language model
======================================
Here includes the source code of Trans-dimensional random field language model (TRFLM), and the scripts of experiment on several dataset, such PTB corpus.

Use `make all` to complie the code and the executables are generated in **bin/**

The package is organized as follows:
- **src/** includes the C++ source code
- **exp/** includes	the scripts and data of the language modeling experiments
  - **exp/Word/** is a word morphology expeirment, a pilot expeirment to modeling English words (character sequences)
  - **exp/PTB/** is the LM experments on Penn Treebank (PTB) dataset. 
  - **exp/Word/data/** and **exp/PTB/data/** contain all the training, valid and test data for LM training. 
    * **\*.lext** is the vocabulary for ngram LMs;
    * **\*.list** is the vocabulary used by TRFLMs. The only difference between **\*.lext** and **\*.list** is that **\*.list** removes the begin and end symbols '\<s\>' and' \</s\>'. 'c10' or 'c200' in the file name denote the class number is 10 or 200.
    * **\*.no** is the dataset corresponding to the vocabulary **\*.lext**, used by ngram LMs
    * **\*.id** is the dataset corresponding to the vocabulary **\*.list**, used by TRFLMs
  - **exp/Word/ngram/** and **exp/PTB/ngram/** contain the python scripts for ngram LMs
  - **exp/Word/trf/** and **exp/PTB/trf/** contain the python scripts for TRFLM training

For a detial introduction of TRFLM, see 
> Wang, Bin, Zhijian Ou, and Zhiqiang Tan. "Trans-dimensional random fields for language modeling." ACL,2015.

For more speech recognition experiments, see https://arxiv.org/abs/1603.09170

