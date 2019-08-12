import torch

class Glove:
    def __init__(self, glove_file_path, max_result_length=10, glove_type='glove6b'):
        self.glove_file_path = glove_file_path
        self.index_to_vector = list()
        self.word_to_index = dict()
        self.vecs_lenght = 0
        self.max_result_length = max_result_length
        num_words = 1917494
        # num_words = 400000

        sum_vec = None
        for ix, (line) in enumerate(open(self.glove_file_path)):
            line = line.split()
            word = line[0]
            vec = [ float(element) for element in line[1:] ]
            vec = torch.tensor( vec )
            if self.vecs_lenght < vec.size()[0]:
                self.vecs_lenght = vec.size()[0]
            self.word_to_index[word] = ix 
            self.index_to_vector.append( vec )
            if sum_vec is None:
                sum_vec = vec
            else:
                sum_vec = torch.add(sum_vec, vec)

            print('%d/%d ( %.3f %% )' % (ix+1, num_words, ((ix+1)/num_words)*100) , end='\r')
        print()
        self.avg_vec = torch.div( sum_vec, len(self.index_to_vector) )

    def __call__(self, line):
        assert type(line) in [int, str, list], 'Cannot use Glove transform for {} type'.format( type(line) ) 
        
        statements = line
        if type(line)==int:
            statements = str(line)
        elif type(line)==list:
            statements = ', '.join( [ '_'.join(statement) if type(statement)==list else str(statement) for statement in line] )
        
        statements = [ statement.split()[0].split('_') for statement in statements.split(',') ]

        encoded_statements = list()

        for ix, (statement) in enumerate(statements):
            encoded_statement = torch.zeros( (len(statement), self.vecs_lenght) )
            for jx, (word) in enumerate(statement):
                encoded_word = None
                if word in self.word_to_index:
                    encoded_word = self.index_to_vector[ self.word_to_index[word] ]
                else:
                    encoded_word = self.avg_vec
                encoded_statement[ jx, :encoded_word.size()[0] ] = encoded_word
            encoded_statements.append( torch.mean( encoded_statement, dim=0 ) )

        encoded_statements_tensor = torch.zeros( (len(encoded_statements), self.avg_vec.size()[0]) )
        for ix, (encoded_statement) in enumerate(encoded_statements):
            encoded_statements_tensor[ix, :] = torch.tensor(encoded_statement)

        return encoded_statements_tensor
