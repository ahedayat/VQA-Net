#!/bin/bash

analysis_num=$1

#### Analysis Parameters ####
num_epochs=50
batch_size=32

#### Data Loader Parameters ####
num_workers=2

#### Optimizer Parameters ####
optimizer='adam'
learning_rate=0.0000001
weight_decay=0.005

#### RNN Parameters ####
rnn_type="GRU"
hidden_size=150
num_layers=1
rnn_version=1

#### Image Encoder Parameters ####
image_encoder="resnet18"

#### Word2Vec Parameters ####
word2vec="glove6b"

python vqa_eval.py  --analysis_num $analysis_num \
                    --epochs $num_epochs\
                    --batch-size $batch_size \
                    --optimization $optimizer \
                    --num_workers $num_workers \
                    --learning-rate $learning_rate \
                    --weight_decay $weight_decay \
                    --rnn_type $rnn_type \
                    --hidden_size $hidden_size \
                    --num_layers $num_layers \
                    --rnn_version $rnn_version \
                    --image-encoder $image_encoder \
                    --word2vec $word2vec \
                    --gpu 
