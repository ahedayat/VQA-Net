import torch
import matplotlib.pyplot as plt

def get_losses(report_root, mode, start_epoch, end_epoch):
    losses_dir = '{}/{}_losses'.format( report_root, mode )
    epochs_losses = list()

    for epoch in range(start_epoch, end_epoch+1):
        epochs_losses.append( torch.load( '{}/{}_losses_epoch_{}.pt'.format( losses_dir, mode, epoch ) ) )

    return epochs_losses

def get_accuracies(report_root, mode, start_epoch, end_epoch):
    accuracies_dir = '{}/{}_accuracies'.format( report_root, mode )
    epochs_accuracies = list()

    for epoch in range(start_epoch, end_epoch+1):
        epochs_accuracies.append( torch.load( '{}/{}_accuracies_epoch_{}.pt'.format( accuracies_dir, mode, epoch ) )*100 )

    return epochs_accuracies

def get_batches_size(report_root, mode, start_epoch, end_epoch):
    batches_size_dir = '{}/{}_batches_size'.format( report_root, mode )
    epochs_batches_size = list()

    for epoch in range(start_epoch, end_epoch+1):
        epochs_batches_size.append( torch.load( '{}/{}_batches_size_epoch_{}.pt'.format( batches_size_dir, mode, epoch ) ) )

    return epochs_batches_size

def get_mean(tensor_list):
    mean_list = list()
    for tensor in tensor_list:
        mean_list.append( torch.mean( tensor ) )
    
    return mean_list

def plot_tensor_list(tensor_list1, tensor_list2, ylabel, xlabel='epochs', q=(1,None) ):
    assert tensor_list2==None or len(tensor_list1)==len(tensor_list2), 'length of two lists must be the same.'

    q1, q2 = q

    epochs = [ix for ix in range( len(tensor_list1) ) ]
    
    list1_mean = get_mean( tensor_list1 )
    plt.plot( epochs, list1_mean, label='Analysis {}'.format(q1) )

    if tensor_list2!=None:
        list2_mean = get_mean( tensor_list2 )
        plt.plot( epochs, list2_mean, label='Analysis {}'.format(q2) )
    
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    plt.xticks( range(0, len(epochs), 1) , range(1, len(epochs)+1, 1) )
    plt.legend(bbox_to_anchor=(1.02,0.5), loc='upper left', borderaxespad=0.0)
    plt.grid()
    plt.show()

def compare(reports_root, start_epoch, end_epoch, q1, q2=None):
    q1_train_losses = get_losses( '{}/{}'.format( reports_root, q1), 'train', start_epoch, end_epoch)
    q1_val_losses = get_losses( '{}/{}'.format( reports_root, q1), 'val', start_epoch, end_epoch)
    q1_test_losses = get_losses( '{}/{}'.format( reports_root, q1), 'test', start_epoch, end_epoch)

    q1_train_accuracies = get_accuracies( '{}/{}'.format( reports_root, q1), 'train', start_epoch, end_epoch)
    q1_val_accuracies = get_accuracies( '{}/{}'.format( reports_root, q1), 'val', start_epoch, end_epoch)
    q1_test_accuracies = get_accuracies( '{}/{}'.format( reports_root, q1), 'test', start_epoch, end_epoch)

    q1_batches_size = get_batches_size( '{}/{}'.format( reports_root, q1), 'train', start_epoch, end_epoch)

    q2_train_losses, q2_val_losses, q2_test_losses = None, None, None
    q2_train_accuracies, q2_val_accuracies, q2_test_accuracies = None, None, None
    q2_batches_size = None

    if q2 is not None:
        q2_train_losses = get_losses( '{}/{}'.format( reports_root, q2), 'train', start_epoch, end_epoch)
        q2_val_losses = get_losses( '{}/{}'.format( reports_root, q2), 'val', start_epoch, end_epoch)
        q2_test_losses = get_losses( '{}/{}'.format( reports_root, q2), 'test', start_epoch, end_epoch)

        q2_train_accuracies = get_accuracies( '{}/{}'.format( reports_root, q2), 'train', start_epoch, end_epoch)
        q2_val_accuracies = get_accuracies( '{}/{}'.format( reports_root, q2), 'val', start_epoch, end_epoch)
        q2_test_accuracies = get_accuracies( '{}/{}'.format( reports_root, q2), 'test', start_epoch, end_epoch)

        q2_batches_size = get_batches_size( '{}/{}'.format( reports_root, q2), 'train', start_epoch, end_epoch)

    plot_tensor_list(q1_train_losses, q2_train_losses, 'Training Loss ( Cross Entropy )', xlabel='epochs', q=(q1,q2) )
    plot_tensor_list(q1_val_losses, q2_val_losses, 'Validation Loss ( Cross Entropy )', xlabel='epochs', q=(q1,q2) )
    plot_tensor_list(q1_test_losses, q2_test_losses, 'Testing Loss ( Cross Entropy )', xlabel='epochs', q=(q1,q2) )

    plot_tensor_list(q1_train_accuracies, q2_train_accuracies, 'Training Accuracy (%)', xlabel='epochs', q=(q1,q2) )
    plot_tensor_list(q1_val_accuracies, q2_val_accuracies, 'Validation Accuracy (%)', xlabel='epochs', q=(q1,q2) )
    plot_tensor_list(q1_test_accuracies, q2_test_accuracies, 'Testing Accuracy (%)', xlabel='epochs', q=(q1,q2) )

def _main():
    reports_root = './reports'
    start_epoch, end_epoch = ( 0, 49 )

    q1, q2 = (1,4)

    compare( reports_root, start_epoch, end_epoch, q1, q2=q2)

if __name__ == "__main__":
    _main()