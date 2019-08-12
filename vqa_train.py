import os
import torch
import warnings

import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils as utility

import transforms as transforms
import torchvision.transforms as torch_transforms
import torchvision.models as torch_models
import dataloader.daquar as daquar
import nets.VQANet as vqanet

def _main(args):
    warnings.filterwarnings("ignore") 

    if args.rnn_version=='1':
        import nets.VQANet as vqanet
    elif args.rnn_version=='2':
        import nets.VQANetv2 as vqanet

    #### Preparing Dataset ####
    datasets_root = './encoded_datasets'
    dataset_name = 'DAQUAR'
    dataset_encoding_version = args.word2vec
    data_mode = 'train'
    dataset_root = '{}/{}_{}/{}'.format( datasets_root, dataset_name, dataset_encoding_version, data_mode)
    net_name = args.image_encoder
    net_out_size = 1024 if args.image_encoder=='densenet121' else 512
    images_root = '{}/images_{}'.format( datasets_root, net_name)
    dataloader = daquar.loader( dataset_root, images_root )

    #### Preparing Pytorch ####
    torch.manual_seed(0)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    #### Constructing Model ####
    rnn_type, rnn_num_layers, rnn_hidden_size, rnn_bidirectional, rnn_dropout = (args.rnn_type, args.rnn_num_layers, args.rnn_hidden_size, args.rnn_bidirectional, False)
    question_word_size = 300
    model = vqanet.model(
                        rnn_type=rnn_type,
                        rnn_hidden_size=rnn_hidden_size,
                        question_word_size=question_word_size,
                        rnn_num_layers=rnn_num_layers,
                        rnn_bidirectional=rnn_bidirectional,
                        rnn_dropout=rnn_dropout,
                        cnn_output_size = net_out_size
                        )
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()

    #### Constructing Criterion ####
    criterion = nn.CrossEntropyLoss()

    #### Constructing Optimizer ####
    optimizer = None
    if args.optimization=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimization=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    #### Training Parameters ####
    start_epoch, num_epoch = (args.start_epoch, args.epochs)
    batch_size = args.batch_size
    num_workers = args.num_workers
    check_counter = 10

    #### Reports Address ####
    reports_root = './reports'
    analysis_num = args.analysis
    reports_path = '{}/{}'.format( reports_root, analysis_num)
    saving_model_path = '{}/models'.format( reports_path )
    model_name = 'vqanet_{}_{}_{}'.format( net_name, start_epoch, start_epoch+num_epoch)

    utility.mkdir( reports_path, 'models', forced_remove=False)
 
    #### Training Model ####
    vqanet.train( 
                vqanet=model,
                train_data=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                num_epoch=num_epoch,
                start_epoch=start_epoch,
                batch_size=batch_size,
                num_workers=num_workers,
                check_counter=check_counter,
                gpu=args.gpu and torch.cuda.is_available(),
                report_path=reports_path,
                saving_model_every_epoch=True
                )
    
    #### Saving Model ####
    vqanet.save(saving_model_path, model_name, model, optimizer=optimizer)

if __name__ == "__main__":
    args = utility.get_args()
    _main(args)
