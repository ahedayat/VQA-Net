import os
import shutil
import torch

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class DaquarImageLoader(Dataset):
	def __init__(self, images_root, images_filename):
		super(DaquarImageLoader, self).__init__()
		self.images_root = images_root
		self.images_filename = images_filename
		self.to_tensor = transforms.ToTensor()

	def __getitem__(self, ix):
		image_filename = self.images_filename[ix]
		image = Image.open( '{}/{}'.format(self.images_root, image_filename) )
		image = self.to_tensor( image )
		return image, image_filename

	def __len__(self):
		return len( self.images_filename )

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
		f.close()
def copy_file(src_path, src_file_name, dst_path, dst_file_name):
	shutil.copyfile('{}/{}'.format(src_path, src_file_name), '{}/{}'.format(dst_path,dst_file_name))  

def ls(dir_path):
	return os.listdir(dir_path)

def preprocess(data_root, data_mode, data_part=(0.,1.)):
	assert data_mode in ['train', 'val', 'test'], 'data mode must be \'train\', \'val\', or \'test\'.'

	mkdir(data_root, data_mode)
	mkdir('{}/{}'.format(data_root, data_mode), 'questions')
	mkdir('{}/{}'.format(data_root, data_mode), 'answers')
	mkdir('{}/{}'.format(data_root, data_mode), 'images')
	touch('{}/{}'.format(data_root, data_mode), 'informations.txt')

	questions_file_name = '{}_questions.txt'.format(data_mode if data_mode!='val' else 'train')
	images_file_name = '{}_images_name.txt'.format(data_mode if data_mode!='val' else 'train')

	questions = [ line.split('\n')[0] for line in open('{}/{}'.format(data_root, questions_file_name)) ]

	answers = questions[1::2]
	questions = questions[0::2]

	num_file_questions = len(questions)
	start_question_index = int(num_file_questions * data_part[0])
	end_question_index = int(num_file_questions * data_part[1])
	num_questions = end_question_index - start_question_index

	for ix, (question, answer) in enumerate( zip( questions[start_question_index:end_question_index], answers[start_question_index:end_question_index]) ):
		
		current_question_name = '{}.txt'.format(ix)
		question = question.split()
		
		# print(answer)
		image_name = question[-2]
		question[-2] = 'image'
		question = ','.join(question)
		question_information ='{} {}'.format(current_question_name, image_name)

		write_file( '{}/{}/questions'.format(data_root, data_mode), current_question_name, question, new_line=False)
		write_file( '{}/{}/answers'.format(data_root, data_mode), current_question_name, answer, new_line=False)
		write_file( '{}/{}'.format(data_root, data_mode), 'informations.txt',  question_information, new_line=True)

		print('%s: %d/%d ( %.2f %% )' % (data_mode, ix+1, num_questions, ( (ix+1)/num_questions )*100 ), end='\r')
	
	copy_file(data_root, images_file_name, '{}/{}'.format(data_root, data_mode), 'images.txt')
	print()

def save_image_representation(cnn_name, images_loader, saving_path_root):
	saving_path = '{}/images_representation_{}'.format( saving_path_root, cnn_name )
	mkdir('{}'.format(saving_path_root), 'images_representation_{}'.format( cnn_name ))

	cnn=None

	if cnn_name=='resnet18':
		cnn = models.resnet18(pretrained=True)
		cnn.fc = nn.Sequential()
	elif cnn_name=='resnet34':
		cnn = models.resnet34(pretrained=True)
		cnn.fc = nn.Sequential()
	elif cnn_name=='densenet121':
		cnn = models.densenet121(pretrained=True)
		cnn.classifier = nn.Sequential()


	dataloader = DataLoader(images_loader,
				batch_size=1,
				shuffle=False,
				pin_memory= False and torch.cuda.is_available(),
				num_workers=2)	
	
	num_images = len(images_loader)
	for ix, (image, image_filename) in enumerate(dataloader):
		image_rep = cnn( image )
		
		for jx in range(image.size()[0]):
			if os.path.splitext(image_filename[jx])[1]!='.png':
				continue
			torch.save( image_rep[jx,:], '{}/{}.pt'.format(saving_path, os.path.splitext(image_filename[jx])[0] ) )

		print('images representation: %d/%d ( %.2f%% )' % (ix+1, num_images, ( (ix+1)/num_images )*100 ), end='\r')
	print()

def _main():
	data_root = '.'

	preprocess(data_root, 'train', data_part=(0.,0.80) )
	preprocess(data_root, 'val', data_part=(0.80,1.) )
	preprocess(data_root, 'test', data_part=(0.,1.) )

	images_root = './images'
	cnn_name = 'densenet121'
	images_root_filenames = ls(images_root)
	images_filename = list()
	for images_root_filename in images_root_filenames:
		if os.path.splitext( images_root_filename )[1]=='.png':
			images_filename.append( images_root_filename )

	dataloader = DaquarImageLoader(images_root, images_filename)

	save_image_representation('resnet18', dataloader, '.')
	save_image_representation('resnet34', dataloader, '.')
	save_image_representation('densenet121', dataloader, '.')

if __name__ == "__main__":
	_main()
