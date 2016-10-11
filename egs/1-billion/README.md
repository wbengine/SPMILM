 
perform LM experiments on Google 1-billion dataset

run_trf.py              training TRF LMs
run_trf_1/2/4.py        training TRF LMs for three corpus separately
run_ngram.py            training ngram LMs
run_trf_sams.py         run SAMS to estimate zeta, fixed lambda
run_rnn.py              training RNN LMs
run_lstm.py             training LSMT LMs
data.py                 prepare the data
run_cmb_wer.py          calculate the WER to interpolate LMs
*.fs                    different feature types
data/1-billion          the 1-billion directory, which should be linked to the 1-billion root directory
data/WSJ92-test-data    the wsj92 1000-best list

-------------------------
Usage:

1. link data/1-billion to the root dir of 1-billion dataset

2. prepare the training data

python data.py

the data will be outputed to three folder 1/2/4

3. train different models

python run_trf/rnn/lstm/ngram.py -all

the PPL and WER results are written to "result.txt"

4. interpolate different LMs

python run_cmb_wer.py

the results are writted to "wer.log"