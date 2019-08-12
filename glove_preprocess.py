import os
import shutil
import re
import warnings
import torch

import numpy as np
import transforms as transforms
import torchvision.models as models
import torchvision.transforms as torch_transforms
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset

def mkdir(dir_path, dir_name, forced_remove=False):
	new_dir = '{}/{}'.format(dir_path,dir_name)
	if forced_remove and os.path.isdir( new_dir ):
		shutil.rmtree( new_dir )
	if not os.path.isdir( new_dir ):
		os.makedirs( new_dir )

def touch(file_path, file_name, forced_remove=False):
	new_file = '{}/{}'.format(file_path,file_name)
	assert os.path.isdir( file_path ), ' \"{}\" does not exist.'.format(file_path)
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

def copy_file(src_path, src_file_name, dst_path, dst_file_name):
	shutil.copyfile('{}/{}'.format(src_path, src_file_name), '{}/{}'.format(dst_path,dst_file_name))  

def copy_dir(src_dir, dst_dir, symlinks=False, ignore=None):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def ls(dir_path):
	return os.listdir(dir_path)

def read_line(file_path, line_num):
    return [ line for line in open(file_path) ][line_num]

def get_all_words(string, spliting_format):
    return re.split(spliting_format, string)

def glove_preprocess(raw_dataset_root, encoded_dataset_root, data_mode, glove):
    assert data_mode in ['train', 'val', 'test'], 'data mode must be \'train\', \'val\' or \'test\'.'

    mkdir(encoded_dataset_root, data_mode)
    mkdir( '{}/{}'.format(encoded_dataset_root, data_mode), 'questions')
    mkdir( '{}/{}'.format(encoded_dataset_root, data_mode), 'answers')
    copy_file( '{}/{}'.format(raw_dataset_root, data_mode), 'images.txt', '{}/{}'.format(encoded_dataset_root, data_mode), 'images.txt')
    copy_file( '{}/{}'.format(raw_dataset_root, data_mode), 'informations.txt', '{}/{}'.format(encoded_dataset_root, data_mode), 'informations.txt')
    
    questions_info = [ (line.split()[0], line.split()[1]) for line in open( '{}/{}/informations.txt'.format( encoded_dataset_root, data_mode) ) ]
    num_questions = len(questions_info)

    all_words = list()
    word_encoding_size = None
    max_question_length = 0

    for ix,(question_info) in enumerate(questions_info):
        question_file_name, question_image = question_info
        question = read_line( '{}/{}/questions/{}'.format( raw_dataset_root, data_mode, question_file_name), 0)
        answers = read_line( '{}/{}/answers/{}'.format( raw_dataset_root, data_mode, question_file_name), 0)

        coded_question = glove(question)
        coded_answers = glove(answers)

        if max_question_length < coded_question.size()[0]:
            max_question_length = coded_question.size()[0]

        question_words = get_all_words(question, ',')
        answers_words = get_all_words(answers, ', |_')

        for question_word in question_words:
            if question_word not in all_words:
                all_words.append( question_word )
        for answer_word in answers_words:
            if answer_word not in all_words:
                all_words.append( answer_word )
        if word_encoding_size is None:
            word_encoding_size = coded_question.size()[1]

        torch.save( coded_question,
                    '{}/{}/questions/{}.pt'.format(encoded_dataset_root, 
                                                data_mode,
                                                os.path.splitext(question_file_name)[0]) )

        torch.save( coded_answers,
                '{}/{}/answers/{}.pt'.format( encoded_dataset_root, 
                                                data_mode,
                                                os.path.splitext(question_file_name)[0] ) )

        print( '%s : %d/%d( %.2f%% )' % ( data_mode, ix+1, num_questions, ( (ix+1)/num_questions )*100 ), end='\r')
    print()
    
    all_words_tensor = torch.zeros( (len(all_words), word_encoding_size) )
    for ix,(word) in enumerate(all_words):
        all_words_tensor[ix, :] = glove(word)
        print('%s (genrating vacabulary tensor): %d/%d( %.2f%% )' % ( data_mode, ix+1, len(all_words), ( (ix+1)/len(all_words) )*100 ), end='\r')
    print()
    
    torch.save( all_words_tensor, 
                '{}/{}/all_words.pt'.format(encoded_dataset_root, data_mode) )

    write_file( '{}/{}'.format(encoded_dataset_root, data_mode), 
                'max_question_length.txt', 
                '{}'.format(max_question_length),
                new_line=False,
                forced_remove_prev=True)

def copy_image_representation(raw_dataset_root, coded_datasets_root , cnn_name):
    images_rep_root = '{}/images_representation_{}'.format(raw_dataset_root, cnn_name)
    image_rep_saving_path = '{}/images_{}'.format( coded_datasets_root, cnn_name)
    copy_dir( images_rep_root, image_rep_saving_path )

def _main():
    warnings.filterwarnings("ignore") 

    dataset_name = 'DAQUAR'
    raw_dataset_root = './datasets/{}'.format(dataset_name)
    coded_datasets = './encoded_datasets'

    embedding_word_type =  'glove6b'
    embedded_words_root = './word_embeding'
    embedded_words_file_name = 'glove_6B_300d.txt'
    embedded_words_dir = '{}/{}/{}'.format( embedded_words_root, embedding_word_type, embedded_words_file_name)
    glove6b = transforms.Glove(embedded_words_dir)

    embedding_word_type =  'glove42b'
    embedded_words_root = './word_embeding'
    embedded_words_file_name = 'glove_42B_300d.txt'
    embedded_words_dir = '{}/{}/{}'.format( embedded_words_root, embedding_word_type, embedded_words_file_name)
    glove42b = transforms.Glove(embedded_words_dir)
    
    encoded_dataset_6b_root = '{}/{}_{}'.format( coded_datasets, dataset_name, 'glove6b')
    encoded_dataset_42b_root = '{}/{}_{}'.format( coded_datasets, dataset_name, 'glove42b')

    glove_preprocess(raw_dataset_root, encoded_dataset_6b_root, 'train', glove6b)
    glove_preprocess(raw_dataset_root, encoded_dataset_6b_root, 'val', glove6b)
    glove_preprocess(raw_dataset_root, encoded_dataset_6b_root, 'test', glove6b)
    
    glove_preprocess(raw_dataset_root, encoded_dataset_42b_root, 'train', glove42b)
    glove_preprocess(raw_dataset_root, encoded_dataset_42b_root, 'val', glove42b)
    glove_preprocess(raw_dataset_root, encoded_dataset_42b_root, 'test', glove42b)

    copy_image_representation(raw_dataset_root, coded_datasets, 'resnet18' )
    copy_image_representation(raw_dataset_root, coded_datasets, 'resnet34' )
    copy_image_representation(raw_dataset_root, coded_datasets, 'densenet121' )
    

if __name__ == "__main__":
    _main()