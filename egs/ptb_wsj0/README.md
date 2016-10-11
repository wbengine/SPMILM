 
Perform the PTB-WSJ experiments.

## python trancripts 
multiple_run_ais.py      run ais for a fixed TRF model several times
multiple_run_ais_1.py    run ais for 10 different TRF models
multiple_run_trf.py      independently run TRF several times
multiple_run_rnn.py      independently run rnn several times
run_lstm.py              run LSTM
run_ngram.py             run ngram LMs
run_rnn.py               run RNN LMs
run_trf.py               run TRF LMs
wer.py                   computer WER for interplated models\
## TRF feature files
g4_w_c_ws_cs_cpw.fs             defination of w+c+ws+cs+cpw
g4_w_c_ws_cs_wsh_csh.fs         defination of w+c+ws+cs+wsh+csh
g4_w_c_ws_cs_wsh_csh_tied.fs    defination of w+c+ws+cs+wsh+csh+tied
## floders
data                    LM corpus and WSJ 1000-best list


---------------------------------------------
Usage:
1. train a single LMs

python run_*.py -all

"run_*.py" denotes run_lstm.py or run_rnn.py or run_ngram.py or run_trf.py
the PPL and WER results are written in the file "model_ppl.txt"

2. if needed, multiple run TRF several times

python multiple_run_trf.py -all

the results will be written to "model_ppl.txt"

2. compute the WER for interplated LMs

python wer.py

the result will be writted to "wer.log"


----------------------------------------------
Rules of *.fs files ( the feature type files )
1. "//" denotes the comments, similar to C++ comments
2. "w" denotes word, "c" denotes class and "-" denotes skip. "[]" after them denotes the order. For example:
    w[1:4]              word ngram with order from 1 to 4
    c[1:4]              class ngram with order form 1 to 4
    w[1]-[1]w[1]        the skip-word-2gram with skip 1, i.e. word-skip-word
    c[1]-[2]c[1]        the skip-class-2gram with skip 2, i.e. class-skip-skip-class
Note: for the type including skip "-", only "-" can follow the range-order "[n:m]". For example:
    w[1]-[1:10]w[1]     supported
    w[1:2]-[1]w[1]      unsupported!!
    





