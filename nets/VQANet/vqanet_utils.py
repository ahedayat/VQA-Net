import gc
import torch
import utils as utility

from torch.autograd import Variable as V
from torch.utils.data import DataLoader

def vqanet_save( file_path, file_name, model, optimizer=None ):
    torch.save({ 
        'net_arch' : 'vqanet',
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict() if optimizer is not None else None,
        }, '{}/{}.pth'.format(file_path,file_name) )

def vqanet_load(file_path, file_name, model, optimizer=None):
    check_points = torch.load('{}/{}.pth'.format(file_path,file_name))
    keys = check_points.keys()

    assert ('net_arch' in keys) and ('model' in keys) and ('optimizer' in keys), 'Cannot read this file in address : {}/{}.pth'.format(file_path,file_name)
    assert check_points['net_arch']=='vqanet', 'This file model architecture is not \'vqanet\''
    model.load_state_dict( check_points['model'] )
    if optimizer is not None:
        optimizer.load_state_dict( check_points['optimizer'] )
    return model, optimizer

def vqanet_accuracy( output, target, gpu=False):
    predicted = torch.argmax( output, dim=1 )
    if gpu:
        predicted = predicted.cuda()
    matched = predicted==target
    acc = int(torch.sum( matched )) / matched.size()[0]
    return acc

def vqanet_train(
                 vqanet,
                 train_data,
                 optimizer,
                 criterion,
                 report_path,
                 num_epoch=1,
                 start_epoch=0, 
                 batch_size=2,
                 num_workers = 1,
                 check_counter=20,
                 gpu=False,
                 saving_model_every_epoch=False):

    utility.mkdir( report_path, 'train_batches_size' )
    utility.mkdir( report_path, 'train_losses' )
    utility.mkdir( report_path, 'train_accuracies' )
    utility.mkdir( report_path, 'models' )

    for epoch in range( start_epoch, start_epoch+num_epoch ):

        data_loader = DataLoader( train_data,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory= gpu and torch.cuda.is_available(),
                                num_workers=num_workers)
        
        batches_size = list()
        losses = list()
        accuracies = list()

        curr_loss = 0
        for ix,(image_rep, question, answer, distractors, question_length) in enumerate( data_loader ):
            image_rep, question, answer, distractors, question_length = V(image_rep), V(question), V(answer), V(distractors), V(question_length, requires_grad=False)
            
            if gpu:
                image_rep, question, answer, distractors, question_length  = image_rep.cuda(), question.cuda(), answer.cuda(), distractors.cuda(), question_length.cuda()

            output = vqanet( (image_rep, question, answer, distractors, question_length), train=True )
            target = torch.tensor( [0] * output.size()[0] )
            if gpu:
                target = target.cuda()

            acc = vqanet_accuracy( output, target, gpu=gpu )
            
            loss = criterion( output, target )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prev_loss = curr_loss
            curr_loss = loss.item()

            batches_size.append( output.size()[0] )
            losses.append( curr_loss )
            accuracies.append( acc )

            print( 'epoch=%d, batch=%d(x%d), prev_loss=%.3f, curr_loss=%.3f, delta=%.3f, acc=%.3f%%' % (
                                                                                                        epoch,
                                                                                                        ix,
                                                                                                        output.size()[1],
                                                                                                        prev_loss,
                                                                                                        curr_loss,
                                                                                                        curr_loss-prev_loss,
                                                                                                        acc*100
                                                                                                    ) )
            if ix%check_counter==(check_counter-1):
                # print()
                pass
            
            del image_rep, question, answer, distractors, question_length
            if gpu:            
                torch.cuda.empty_cache()
            gc.collect()

        torch.save( torch.tensor( batches_size ), 
                    '{}/train_batches_size/train_batches_size_epoch_{}.pt'.format(
                                                                                    report_path,
                                                                                    epoch
                                                                                 )
                  )
        torch.save( torch.tensor( losses ), 
                    '{}/train_losses/train_losses_epoch_{}.pt'.format(
                                                                                    report_path,
                                                                                    epoch
                                                                                 )
                  )
        torch.save( torch.tensor( accuracies ), 
                    '{}/train_accuracies/train_accuracies_epoch_{}.pt'.format(
                                                                                    report_path,
                                                                                    epoch
                                                                                 )
                  )
        if saving_model_every_epoch:
            vqanet_save( 
                        '{}/models'.format( report_path ),
                        'vqanet_epoch_{}'.format( epoch ),
                        vqanet,
                        optimizer=optimizer
                       )

def vqanet_eval( 
                vqanet,
                eval_data,
                criterion,
                report_path,
                epoch,
                batch_size=2,
                num_workers=2,
                check_counter=4,
                gpu=False,
                eval_mode='test'):

    assert eval_mode in ['val', 'test'], 'eval mode must be \'val\' or \'test\''

    utility.mkdir(report_path, '{}_batches_size'.format(eval_mode))
    utility.mkdir(report_path, '{}_losses'.format(eval_mode))
    utility.mkdir(report_path, '{}_accuracies'.format(eval_mode))

    data_loader = DataLoader(eval_data,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=gpu and torch.cuda.is_available(),
                num_workers=num_workers)
    vqanet = vqanet.eval()

    batches_size = list()
    losses = list()
    accuracies = list()

    curr_loss = 0
    for ix,(image_rep, question, answer, distractors, question_length) in enumerate( data_loader ):
        image_rep, question, answer, distractors, question_length = V(image_rep), V(question), V(answer), V(distractors), V(question_length, requires_grad=False)
        
        if gpu:
            image_rep, question, answer, distractors, question_length  = image_rep.cuda(), question.cuda(), answer.cuda(), distractors.cuda(), question_length.cuda()
        
        output = vqanet( (image_rep, question, answer, distractors, question_length), train=False )
        target = torch.tensor( [0] * output.size()[0] )
        if gpu:
            target = target.cuda()

        acc = vqanet_accuracy( output, target, gpu=gpu )
        
        loss = criterion(output, target)

        prev_loss = curr_loss
        curr_loss = loss.item()

        batches_size.append( output.size()[0] )
        losses.append( curr_loss )
        accuracies.append( acc )


        print( 'batch=%d(x%d), prev_loss=%.3f, curr_loss=%.3f, delta=%.3f, acc=%.3f%%' % (
                                                                                            ix,
                                                                                            output.size()[1],
                                                                                            prev_loss,
                                                                                            curr_loss,
                                                                                            curr_loss-prev_loss,
                                                                                            acc*100
                                                                                          ) )
        if ix%check_counter==(check_counter-1):
            # print()
            pass
            
        del image_rep, question, answer, distractors, question_length
        if gpu:            
            torch.cuda.empty_cache()
        gc.collect()

        torch.save( torch.tensor( batches_size ), 
                    '{}/{}_batches_size/{}_batches_size_epoch_{}.pt'.format(
                                                                    report_path,
                                                                    eval_mode,
                                                                    eval_mode,
                                                                    epoch
                                                                  )
                  )
        torch.save( torch.tensor( losses ), 
                    '{}/{}_losses/{}_losses_epoch_{}.pt'.format(
                                                        report_path,
                                                        eval_mode,
                                                        eval_mode,
                                                        epoch
                                                      )
                  )
        torch.save( torch.tensor( accuracies ), 
                    '{}/{}_accuracies/{}_accuracies_epoch_{}.pt'.format(
                                                                report_path,
                                                                eval_mode,
                                                                eval_mode,
                                                                epoch
                                                              )
                  )

def change_view(tensor, new_view):
    for dim in range( len(tensor.size()) ):
        assert dim in new_view, 'dim={} does not exist in new_view'.format(dim)

    tensor_size = tensor.size()
    return tensor.reshape( tensor_size[new_view[0]], tensor_size[new_view[1]], tensor_size[new_view[2]])
