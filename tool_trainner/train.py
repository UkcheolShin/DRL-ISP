import argparse
import os
import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from imageio import imread, imwrite # rgb read
from torch.optim import lr_scheduler

import sys # load img modifer
sys.path.append("..")
from common.CNN_tool_v2 import CNN_TOOL_V2

# data loader
from torch.utils.data.dataloader import DataLoader
from data_loader.loader_individual import DataSetLoader
from data_loader import custom_transforms

# Loss
from loss_fn.loss_functions import SSIM, tv_loss
from loss_fn.vgg_loss import VGGPerceptualLoss

# utils
from utils import AverageMeter, calc_psnr, bayer_to_rgb
from logger import Logger, Printer
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1023)
    parser.add_argument('--log_dir', type=str, default='logs/', help='Path for logs')
    parser.add_argument('--dataset_dir', type=str, default='/SSD1/Dataset_DRLISP/', required=True)

    # dataset / training target option
    parser.add_argument('--dataset', type=str, default='syn', choices=['ETH', 'syn', 'wb'])
    parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'noise', 'gusblur', 'jpeg', 'bright_gam', 'sr', 'wb'], required=True)
    parser.add_argument('--level', type=str, default='low', choices=['low', 'high'], required=True)
    parser.add_argument('--augmentation', action='store_true', help='')

    # Network option
    parser.add_argument('--num-channels', type=int, default=4)
    parser.add_argument('--network', type=str, default='CNN_TOOL3')
    parser.add_argument('--color', type=str, default=None, choices=['hsv', 'xyz', 'hls', 'luv', 'ycbcr', 'yuv'])
    parser.add_argument('--net_skip', action='store_true', help='Whether to save results')

    parser.add_argument('--l1', action='store_true', help='Whether to save results')
    parser.add_argument('--mse', action='store_true', help='Whether to save results')
    parser.add_argument('--ssim', action='store_true', help='Whether to save results')
    parser.add_argument('--vgg_feat', action='store_true', help='Whether to save results')
    parser.add_argument('--tv', action='store_true', help='Whether to save results')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. create log & output folder
    log_folder = os.path.join(args.log_dir, args.outputs_dir[8:], args.network)
    args.outputs_dir = os.path.join(args.outputs_dir,'{}'.format(args.network))
    os.makedirs(log_folder, exist_ok=True)

    logger = Logger(log_folder)
    logger.log(json.dumps(vars(args), indent=4))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # 2. load dataloader
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        # custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
    ])
    valid_transform  = custom_transforms.Compose([custom_transforms.ArrayToTensor()])

    train_dataset = DataSetLoader(dataset_dir=args.dataset_dir, 
                                  dataset = args.dataset,
                                  mode = args.mode,
                                  aug = args.augmentation,
                                  tf  = train_transform,
                                  level=args.level,
                                  is_train = True)
    eval_dataset  = DataSetLoader(dataset_dir=args.dataset_dir, 
                                  dataset = args.dataset,
                                  mode = args.mode,
                                  aug = False,
                                  tf  = valid_transform,
                                  level=args.level,
                                  is_train = False)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataloader = DataLoader(dataset=eval_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)


    # 3. make model & optimizer
    model = CNN_TOOL_V2(num_channels=args.num_channels, converter = args.color, use_skips=args.net_skip, network=args.network).to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': args.lr}], betas=(0.9,0.999), weight_decay=0)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99) 

    compute_ssim_loss = SSIM().to(device)
    loss_MSE = nn.MSELoss()
    loss_L1 = nn.L1Loss()
    loss_VGG = VGGPerceptualLoss(resize=True).to(device)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        epoch_losses_mse = AverageMeter()
        epoch_losses_l1 = AverageMeter()
        epoch_losses_ssim = AverageMeter()
        epoch_losses_vgg = AverageMeter()
        epoch_losses_tv = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = torch.zeros(1).to(device)

                if args.l1 :
                    loss_l1  = (preds - labels).abs().clamp(0,1.).mean()#loss_L1(preds, labels)                
                    loss += loss_l1
                    epoch_losses_l1.update(loss_l1.item(), len(inputs))
                if args.mse :
                    loss_mse = loss_MSE(preds, labels)
                    loss += loss_mse
                    epoch_losses_mse.update(loss_mse.item(), len(inputs))
                if args.vgg_feat :
                    # loss_vgg = 0.4*loss_VGG(preds, labels, feature_layers=[2], style_layers=[0, 1, 2, 3]) # featyre + style recon loss
                    loss_vgg = 10*loss_VGG(preds, labels, feature_layers=[2], style_layers=[]) # featyre recon loss
                    loss += loss_vgg
                    epoch_losses_vgg.update(loss_vgg.item(), len(inputs))
                if args.ssim :
                    loss_ssim = compute_ssim_loss(preds,labels).mean()
                    loss += loss_ssim
                    epoch_losses_ssim.update(loss_ssim.item(), len(inputs))
                if args.tv : 
                    loss_tv   = tv_loss(preds)
                    loss += loss_tv
                    epoch_losses_tv.update(loss_tv.item(), len(inputs))

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        scheduler.step()
        cur_lr = scheduler.get_last_lr()
        logger.summary_scalar('train/lr', cur_lr[0], epoch)
        logger.summary_scalar('train/loss_avg', epoch_losses.avg, epoch)

        if args.l1 :
            logger.summary_scalar('train/loss_l1', epoch_losses_l1.avg, epoch)
        if args.mse :
            logger.summary_scalar('train/loss_mse', epoch_losses_mse.avg, epoch)
        if args.vgg_feat :
            logger.summary_scalar('train/loss_vgg', epoch_losses_vgg.avg, epoch)
        if args.ssim :
            logger.summary_scalar('train/loss_ssim', epoch_losses_ssim.avg, epoch)
        if args.tv : 
            logger.summary_scalar('train/loss_tv', epoch_losses_tv.avg, epoch)

        if epoch % 20 == 0 :
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        if epoch % 20 == 0 :
            result_foler = os.path.join(args.outputs_dir,'{}'.format(epoch))
            if not os.path.exists(result_foler):
                os.makedirs(result_foler)

        idx=0
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                preds = model(inputs)#.clamp(0.0, 1.0)

            if epoch % 20 == 0 :
                input_ = 255*bayer_to_rgb(inputs.squeeze(0).cpu().detach().numpy())
                label_ = 255*bayer_to_rgb(labels.squeeze(0).cpu().detach().numpy())
                pred_ = 255*bayer_to_rgb(preds.squeeze(0).cpu().detach().numpy())
                total = np.concatenate((input_,pred_,label_), axis=2).transpose(1,2,0)
                file_name = result_foler+'/'+str(idx).zfill(5)+'.jpg'
                imwrite(file_name, total.astype(np.uint8))
                idx +=1

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        logger.summary_scalar('test/eval_psnr', epoch_psnr.avg, epoch)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
            torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
