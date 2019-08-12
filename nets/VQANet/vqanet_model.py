import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqanet_utils import change_view

class VQANet(nn.Module):
    def __init__(self, rnn_type,rnn_hidden_size, question_word_size, rnn_num_layers=1, rnn_bidirectional=False, rnn_dropout=False, cnn_output_size=512):
        super(VQANet, self).__init__()
        assert rnn_type in ['GRU', 'LSTM'], 'rnn_type must be GRU or LSTM'

        rnn_output_size =  (2 if rnn_bidirectional else 1) * rnn_hidden_size

        linear_layer_output_size = question_word_size

        self.init_layer_h = nn.Sequential(
                                        nn.Linear(cnn_output_size, rnn_hidden_size, bias=True),
                                        nn.ReLU()
                                       )
        self.init_layer_c = nn.Sequential(
                                        nn.Linear(cnn_output_size, rnn_hidden_size, bias=True),
                                        nn.ReLU()
                                       )  

        self.rnn = None
        self.rnn_type = rnn_type
        self.rnn_h = None
        self.rnn_c = None
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_num_layers = rnn_num_layers
        rnn_dropout = int(rnn_dropout)

        if self.rnn_type=='GRU':
            self.rnn = nn.GRU(
                                input_size = question_word_size,
                                hidden_size = rnn_hidden_size,
                                num_layers = rnn_num_layers,
                                bidirectional = rnn_bidirectional,
                                dropout = rnn_dropout
                             )

        elif self.rnn_type=='LSTM':
            self.rnn = nn.LSTM(
                                input_size = question_word_size,
                                hidden_size = rnn_hidden_size,
                                num_layers = rnn_num_layers,
                                bidirectional = rnn_bidirectional,
                                dropout = rnn_dropout
                              )
            
        self.linear_layer = nn.Sequential(
                                            nn.Linear( rnn_output_size+cnn_output_size, linear_layer_output_size),
                                            nn.ReLU(),
                                            nn.Linear( linear_layer_output_size, linear_layer_output_size),
                                            nn.ReLU()
                                         )
        self.softmax = nn.Softmax2d()
        
    def forward(self, image_question_answer_distractors, train=False):

        image_rep, encoded_question, encoded_answer, encoded_distractors, questions_length = image_question_answer_distractors

        self.initialize_rnn( image_rep )

        image_rep = change_view( image_rep, [1,0,2] )
        encoded_question = change_view( encoded_question, [1,0,2] )
        encoded_answer = change_view( encoded_answer, [1,0,2] )
        encoded_distractors = change_view( encoded_distractors, [1,0,2] )
        
        if self.rnn_type=='GRU':
            encoded_question, self.rnn_h = self.rnn( encoded_question, self.rnn_h )
        else:
            encoded_question, (self.rnn_h, self.rnn_c) = self.rnn( encoded_question, (self.rnn_h, self.rnn_c) )
        
        last_question_word_output = encoded_question[ encoded_question.size()[0]-1, :, : ].unsqueeze(dim=0)

        output = None
        
        if train or not train:
            encoded_question = torch.cat( (image_rep, last_question_word_output ), dim=2 )
            encoded_question = self.linear_layer( encoded_question )
            
            
            encoded_distractors = torch.cat( (encoded_answer, encoded_distractors), dim=0 )

            encoded_question = encoded_question.expand( encoded_distractors.size()[0], encoded_question.size()[1], encoded_question.size()[2] )
            encoded_distractors = torch.einsum('ijk,ijk->ij',[encoded_distractors, encoded_question])
            output = encoded_distractors
        else:
            output = self.softmax( last_question_word_output )

        return output
    
    def initialize_rnn(self, image_rep):
        self.rnn_h = self.init_layer_h(image_rep)
        self.rnn_h = self.rnn_h.reshape( 
                                        self.rnn_h.size()[1], 
                                        self.rnn_h.size()[0], 
                                        self.rnn_h.size()[2] ).contiguous()
        self.rnn_h = self.rnn_h.expand( 
                                        self.rnn_h.size()[0] * self.rnn_num_layers,
                                        self.rnn_h.size()[1],
                                        self.rnn_h.size()[2]
                                      ).contiguous()

        if self.rnn_bidirectional:
            self.rnn_h = self.rnn_h.expand( 2*self.rnn_h.size()[0], self.rnn_h.size()[1], self.rnn_h.size()[2] ).contiguous()

        if self.rnn_type=='LSTM':
            self.rnn_c = self.init_layer_c(image_rep)
            self.rnn_c = self.rnn_c.view( self.rnn_c.size()[1], self.rnn_c.size()[0], self.rnn_c.size()[2] ).contiguous()
            self.rnn_c = self.rnn_c.expand( 
                                self.rnn_c.size()[0] * self.rnn_num_layers,
                                self.rnn_c.size()[1],
                                self.rnn_c.size()[2]
                                ).contiguous()
            if self.rnn_bidirectional:
                self.rnn_c = self.rnn_c.expand( self.rnn_c.size()[0], self.rnn_c.size()[1], self.rnn_c.size()[2] ).contiguous()

        
