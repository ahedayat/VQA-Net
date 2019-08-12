import os
import random
import torch

import numpy as np
import torchvision.transforms as transforms

from random import seed
from random import gauss
from PIL import Image
from torch.utils.data import Dataset
from .daquar_loader_utils import ls

class DAQUARLoader(Dataset):
    """
        data_root/
                 |_questions/
                 |_answers/
                 |_informations.txt
                 |_images.txt
        images_rep_root/
                 |_*.pt
        labels_address:
            Address of file that contain labels
    """
    def __init__(self,  data_root, 
                        images_rep_root, 
                        num_distractors=31, 
                        max_question_size=50,
                        question_transform=None, 
                        answer_transform=None):
        
        self.data_root = data_root
        self.images_rep_root = images_rep_root
        self.questions_path = '{}/questions'.format( self.data_root )
        self.answers_path = '{}/answers'.format( self.data_root )
        self.num_distractors = num_distractors

        # self.informations = [ (line.split()[0], line.split()[1]) for line in open( '{}/informations.txt'.format(self.data_root) ) ]
        self.questions_filename = [ line.split()[0] for line in open( '{}/informations.txt'.format(self.data_root) ) ]
        self.questions_image = [ line.split()[1] for line in open( '{}/informations.txt'.format(self.data_root) ) ]
        self.max_question_length = [ int(line.split()[0]) for line in open('{}/max_question_length.txt'.format( self.data_root ) ) ][0]
        self.all_words = torch.load( '{}/all_words.pt'.format(data_root) )

        self.question_transform = question_transform
        self.answer_transform = answer_transform

    def __getitem__(self, ix):
        question_filename = os.path.splitext(self.questions_filename[ix])[0]
        image_name = self.questions_image[ix]

        image_rep =torch.load( '{}/{}.pt'.format( self.images_rep_root, image_name ) )
        question = torch.load( '{}/{}.pt'.format(self.questions_path, question_filename) )
        answers = torch.load( '{}/{}.pt'.format(self.answers_path, question_filename) )
        
        answer_index = random.randint(0, answers.size()[0]-1)
        answer = answers[ answer_index, :]

        distractors = self.generate_distractor(ix, question, answers, answer_index, distractor_size=answer.size()[0])

        if self.question_transform is not None:
            question = self.question_transform(question)
        if self.answer_transform is not None:
            answer = self.answer_transform(answer)

        image_rep = torch.unsqueeze(image_rep, 0)
        image_rep = image_rep.detach()
        image_rep.requires_grad = False
        answer = torch.unsqueeze(answer, 0)
        padded_qeustion = torch.zeros((self.max_question_length, question.size()[1]))
        padded_qeustion[ :question.size()[0], :] = question

        return image_rep, padded_qeustion, answer, distractors, question.size()[0]

    def __len__(self):
        return len( self.questions_filename )

    def generate_distractor(self, ix, question, answers, answer_index, distractor_size=300, num_question_word_choosing=5):
        distractors = torch.zeros( (self.num_distractors, distractor_size) )

        for ix in range(0,num_question_word_choosing):
            question_index = random.randint(0, question.size()[0]-1)
            distractors[ix, :] = question[ question_index, : ]
        for ix in range(num_question_word_choosing, self.num_distractors):
            word_index = random.randint(0, self.all_words.size()[0]-1)
            distractors[ix, :] = self.all_words[ word_index, : ]

        return distractors
