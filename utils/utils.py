import os
import shutil
import torch
from optparse import OptionParser


def mkdir(dir_path, dir_name, forced_remove=False):
    new_dir = '{}/{}'.format(dir_path, dir_name)
    if forced_remove and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)


def touch(file_path, file_name, forced_remove=False):
    new_file = '{}/{}'.format(file_path, file_name)
    assert os.path.isdir(
        file_path), ' \"{}\" does not exist.'.format(file_path)
    if forced_remove and os.path.isfile(new_file):
        os.remove(new_file)
    if not os.path.isfile(new_file):
        open(new_file, 'a').close()


def write_file(file_path, file_name, content, new_line=True, forced_remove_prev=False):
    touch(file_path, file_name, forced_remove=forced_remove_prev)
    with open('{}/{}'.format(file_path, file_name), 'a') as f:
        f.write('{}'.format(content))
        if new_line:
            f.write('\n')
        f.close()


def copy_file(src_path, src_file_name, dst_path, dst_file_name):
    shutil.copyfile('{}/{}'.format(src_path, src_file_name),
                    '{}/{}'.format(dst_path, dst_file_name))


def ls(dir_path):
    return os.listdir(dir_path)


def get_args():
    parser = OptionParser()
    parser.add_option('-a', '--analysis', dest='analysis', default=1, type='int',
                      help='analysis number')
    parser.add_option('-v', '--validate', action='store_true', dest='val',
                      default=False, help='validate model')

    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=2,
                      type='int', help='batch size')
    parser.add_option('-r', '--learning-rate', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('-m', '--momentum', dest='momentum', default=0.9,
                      type='float', help='momentum for sgd optimizer')
    parser.add_option('-d', '--weight_decay', dest='weight_decay', default=0.005,
                      type='float', help='weight decay for adam optimizer')

    parser.add_option('-o', '--optimization', type='choice', action='store', dest='optimization',
                      choices=('sgd', 'adam'), default='adam', help='optimization method')

    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-w', '--num_workers', dest='num_workers', default=1,
                      type='int', help='use cuda')

    parser.add_option('-s', '--start-epoch', dest='start_epoch', default=0,
                      type='int', help='starting epoch number')

    parser.add_option('--image-encoder', type='choice', action='store', dest='image_encoder',
                      choices=('resnet18', 'resnet34', 'densenet121'), default='resnet18',
                      help='image encoder: [ \'resnet18\',\'resnet34\',\'densenet121\']')
    parser.add_option('--word2vec', type='choice', action='store', dest='word2vec',
                      choices=('glove6b', 'glove42b'), default='glove6b',
                      help='word2vec: [ \'glove6b\',\'glove42b\']')
    parser.add_option('--rnn_type', type='choice', action='store', dest='rnn_type',
                      choices=('GRU', 'LSTM'), default='GRU',
                      help='word2vec: [ \'GRU\',\'LSTM\']')
    parser.add_option('--hidden_size', dest='rnn_hidden_size', default=150, type='int',
                      help='rnn hidden size')
    parser.add_option('--num_layers', dest='rnn_num_layers', default=1, type='int',
                      help='rnn num layers')
    parser.add_option('--bidirectional', action='store_true', dest='rnn_bidirectional',
                      default=False, help='bidirectional rnn')
    parser.add_option('--rnn_version', type='choice', action='store', dest='rnn_version',
                      choices=('1', '2'), default='1',
                      help='rnn version: [ \'1\',\'2\']')

    (options, args) = parser.parse_args()
    return options
