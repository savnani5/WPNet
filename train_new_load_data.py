import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
#from tensorboardX import SummaryWriter
from torchsummary import summary

from pytorch_model_skip import PTModel as Model
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize
#from load_weight_from_keras_pass_to_model import model_pass

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()


    pretrained_dict=torch.load("model_keras.pth")
    # Create model
    model = Model().cuda()
    print('Model created.')
    # print(model)
    # summary(model, (3, 256, 256))

    #load old weights
    
    # state_dict_filt={k[key_:]: v for k,v in checkpoint_file['model'].items()}

    # for k,v in checkpoint_file['model'].items():
    #   if
    model_dict=model.state_dict()
    # model.load_state_dict(model_dict)
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # update_dict={k: v for k, v in pretrained_dict.items() if k in model_dict}
    update_dict={}
    # print(pretrained_dict.keys())
    print(model_dict.keys())
    for k, v in model_dict.items():
      if k in pretrained_dict:
        print('available so uploading that sht',k)
        model_dict[k]=pretrained_dict[k]
      # else:
      #   print('not available so normal one',k)
      #   model_dict={k:v}
    print(model_dict.keys())
    # update_dict={k: pretrained_dict[k] if k in pretrained_dict else k:v for k, v in model_dict.items() }

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(model_dict) 
    # # 3. load the new state dict
    model.load_state_dict(model_dict)

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    #writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()
    #record the loss
    loss_list=[100000]
    
    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output = model(image)

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim) + (0.1 * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            
                # Print to console
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'ETA {eta}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'
            .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
                # torch.save(model.state_dict(), './model_' + str(epoch) + '.pth')
                # print('model saved to: ', './model_' + str(epoch) + '.pth')
                # Log to tensorboard
                #writer.add_scalar('Train/Loss', losses.val, niter)

            # if i == int(len(train_loader)*0.5):
            #   torch.save(model.state_dict(), './model_int_' + str(epoch) + '.pth')
            #   print('half model saved to: ', './model_int_' + str(epoch) + '.pth')

            #     LogProgress(model, writer, test_loader, niter)

        # Record epoch's intermediate results
        #LogProgress(model, writer, test_loader, niter)
        #writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
        test_loss=Testing(model, test_loader)
        print('Test loss: ',test_loss)
        if test_loss<min(loss_list):
          torch.save(model.state_dict(), './model_' + str(epoch) + '.pth')
          print('model saved to: ', './model_' + str(epoch) + '.pth')
        loss_list.append(test_loss)
        torch.save(model.state_dict(), './model_last.pth')


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image) )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output

def Testing(model, test_loader):
    # model.eval()
    loss_tot=0
    l1_criterion = nn.L1Loss()
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):

            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    
            # Normalize depth
            depth_n = DepthNorm( depth )
    
            # Predict
            output = model(image)
    
            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
    
            loss = (1.0 * l_ssim) + (0.1 * l_depth)
            loss_tot=loss_tot+loss
        loss_tot=loss_tot.item() / len(test_loader)
        return loss_tot


if __name__ == '__main__':
    main()
