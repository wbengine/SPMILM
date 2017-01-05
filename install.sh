#!/bin/bash

srilm_dir='tools/srilm'
rnn_dir='tools/rnn/rnnlm-0.3e'
lstm_dir='tools/lstm'
trf_dir='tools/trf'

root_dir=$(pwd)

#install rnn
cd ${root_dir}'/'${rnn_dir}
make all

cd ${root_dir}'/'${trf_dir}
make clean
make all

#install srilm
cd ${root_dir}'/'${srilm_dir}
tar -xzf srilm-1.7.1.tar.gz
mv Makefile Makefile.sv
cat Makefile.sv | sed "/SRILM =/i\SRILM = $(pwd)/" > Makefile
make World
cp $(find bin -name ngram-count) . --force
cp $(find bin -name ngram) . --force


