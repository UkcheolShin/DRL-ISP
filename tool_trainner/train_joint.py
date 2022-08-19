import argparse
import os
import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from imageio import imread, imwrite # rgb read

import sys # load img modifer
sys.path.append("..")
from common.CNN_tool_v2 import CNN_TOOL_V2
from loss_fn.loss_functions import SSIM

from data_loader.loader_joint import DataSetLoader
from utils import AverageMeter, calc_psnr, bayer_to_rgb
from logger import Logger, Printer
import json
torch.autograd.set_detect_anomaly(True)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

L_mse = nn.MSELoss()
L_l1 = nn.L1Loss()
L_ssim = SSIM().to(device)

def get_loss(preds, labels) : 
    loss_l1   = L_l1(preds, labels)
    # loss_mse  = L_mse(preds, labels)
    # loss_ssim = L_ssim(preds, labels).mean()

    loss = loss_l1
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='/SSD1/Dataset_DRLISP/', required=True)
    parser.add_argument('--dataset', type=str, default='ETH', choices=['ETH', 'syn', 'wb'])
    parser.add_argument('--augmentation', action='store_true', help='')
    parser.add_argument('--weight-dir', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--name', type=str, default='logs/', help='Path for logs')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Path for logs')
    parser.add_argument('--net_skip', action='store_true', help='Whether to save results')
    parser.add_argument('--sequence_len', type=int, default=2)
    parser.add_argument('--with-global', action='store_true', help='Whether to save results')
    parser.add_argument('--with-local', action='store_true', help='Whether to save results')

    args = parser.parse_args()

    log_folder = os.path.join(args.log_dir, args.outputs_dir[8:])

    os.makedirs(log_folder, exist_ok=True)
    logger = Logger(log_folder)
    logger.log(json.dumps(vars(args), indent=4))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    # 1. build & load toolbox
    tool_cnn_list = ['DeNoiseNet_L3.pth', 'DeNoiseNet_L8.pth', 'SRNet_L3.pth', 'SRNet_L8.pth', 'DeJpegNet_L3.pth', 'DeJpegNet_L8.pth',\
                     'DeBlurNet_L3.pth', 'DeBlurNet_L8.pth', 'ExposureNet_L3.pth', 'ExposureNet_L8.pth', 'CTCNet_L3.pth', 'CTCNet_L8.pth']
 

    # Add CNN-based tools
    tool_name = []
    tools = {}
    for idx in range(len(tool_cnn_list)):
        if 'L3' in tool_cnn_list[idx] :
            net_type = 'CNN_TOOL3'
        else:
            net_type = 'CNN_TOOL8'

        Net_Tool = CNN_TOOL_V2(num_channels=4, converter = None, use_skips=True, network=net_type).to(device)
        checkpoint = torch.load(args.weight_dir + tool_cnn_list[idx])
        Net_Tool.load_state_dict(checkpoint)

        tools[tool_cnn_list[idx].split('.')[0]] = Net_Tool
        tool_name.append(tool_cnn_list[idx].split('.')[0])

    # 2. define optimizers
    optimizers = {}
    for idx in range(len(tools)):
        optimizer = optim.Adam([
            {'params': tools[tool_name[idx]].parameters(), 'lr': args.lr}
        ])
        optimizers[tool_name[idx]] = optimizer

    # 3. define training & evaluation dataset         
    train_dataset = DataSetLoader(dataset_dir=args.dataset_dir, 
                                 sequence_len=args.sequence_len,
                                 dataset = args.dataset,
                                 aug = args.augmentation,
                                 is_train=True)
    
    eval_dataset = DataSetLoader(dataset_dir=args.dataset_dir, 
                                 sequence_len=args.sequence_len,
                                 dataset = args.dataset,
                                 aug = False,
                                 is_train=False)

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

    for i in range(len(tool_name)) : 
        print('{} | CNN_tool : {} , train : {}, eval : {}'.format(i, tool_name[i], train_dataloader.dataset.actions_name[i], eval_dataloader.dataset.actions_name[i]))

    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        epoch_losses = {}
        for idx in range(len(tools)) : 
            tools[tool_name[idx]].train()
            epoch_losses[tool_name[idx]] = AverageMeter() 

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
        
            for data in train_dataloader:
                images, actions = data
                if args.with_local:
                    # 1. Local loop
                    # B x c x h x w, B x len x c x h x w, B x len (same batches' action are all same.)
                    # restore image in reversed way
                    for idx, action in zip(reversed(range(args.sequence_len)), reversed(actions)) :
                        action = action[0]
                        inputs = images[:,idx+1,:,:,:].float().to(device)
                        labels = images[:,idx,:,:,:].float().to(device)

                        if idx == args.sequence_len-1 :
                            # local graph
                            preds    = tools[action](inputs).clamp(0.0, 1.0)
                            loss_lc  = get_loss(preds, labels)
                        else:
                            # local graph
                            preds_1  = tools[action](inputs).clamp(0.0, 1.0)
                            preds_2  = tools[action](pred_prev).clamp(0.0, 1.0)

                            loss_lc1  = get_loss(preds_1, labels)
                            loss_lc2  = get_loss(preds_2, labels)
                            loss_lc = loss_lc1 + loss_lc2

                        epoch_losses[action].update(loss_lc.item(), len(inputs))
                        pred_prev = preds.detach().clone() # detach the graph, make independent graph per every step.
                        
                        # per 
                        optimizers[action].zero_grad()
                        loss_lc.backward()                
                        optimizers[action].step()

                if args.with_global:
                    # 2. Global loop
                    b, num_a, c, h, w = images.shape
                    inputs = images[:,-1,:,:,:].float().to(device) # Initial image
                    labels = images[:,0,:,:,:].float().to(device)

                    for action in reversed(actions) :
                        action = action[0]
                        pred = tools[action](inputs).clamp(0.0, 1.0)
                        inputs = pred 

                    # global loss propagation
                    loss_gb = get_loss(pred,labels)

                    for action in actions:
                        action = action[0]
                        optimizers[action].zero_grad()

                    loss_gb.backward()                

                    for action in actions:
                        action = action[0]
                        optimizers[action].step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses[tool_name[idx]].avg))
                t.update(len(inputs))

                # re-set actions per batch
                train_dataloader.dataset.set_actions()

        # Logging & set evaluation mode
        for i in range(len(tools)):
            logger.summary_scalar('train/loss_avg_' + tool_name[i], epoch_losses[tool_name[i]].avg, epoch)
            tools[tool_name[i]].eval()

        if epoch % 20 == 0 :
            # save weight file 
            for idx in range(len(tools)):
                torch.save(tools[tool_name[idx]].state_dict(), os.path.join(args.outputs_dir, '{}_{}.pth'.format(epoch, tool_name[idx])))

            # make result dir 
            result_foler = os.path.join(args.outputs_dir,'{}'.format(epoch))
            if not os.path.exists(result_foler):
                os.makedirs(result_foler)

        epoch_psnr = AverageMeter()

        idx=0
        for data in eval_dataloader:
            images, actions = data
            inputs = images[:,-1,:,:,:].float().to(device) # Initial image
            labels = images[:,0,:,:,:].float().to(device)

            iddx=0
            with torch.no_grad():
                inputs_ = inputs
                for i, act in zip(reversed(range(len(actions))), reversed(actions)) :
                    act = act[0] # I don't know why, data loader automatically convert str to tuple. 
                    preds = tools[act](inputs_)#.clamp(0.0, 1.0)

                    if epoch % 20 == 0 :
                        input_ = 255*bayer_to_rgb(inputs_.squeeze(0).cpu().detach().numpy())
                        label_ = 255*bayer_to_rgb(images[:,i,:,:,:].squeeze(0).cpu().detach().numpy())
                        pred_ = 255*bayer_to_rgb(preds.squeeze(0).cpu().detach().numpy())
                        total = np.concatenate((input_,pred_,label_), axis=2).transpose(1,2,0)
                        file_name = result_foler+'/'+str(idx).zfill(5)+'_'+str(iddx).zfill(2)+'_'+act+'.jpg'
                        imwrite(file_name, total.astype(np.uint8))
                        iddx += 1

                    inputs_ = preds

            idx +=1

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            eval_dataloader.dataset.set_actions()
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        logger.summary_scalar('test/eval_psnr', epoch_psnr.avg, epoch)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            for idx in range(len(tools)):
                best_weights = copy.deepcopy(tools[tool_name[idx]].state_dict())
                torch.save(best_weights, os.path.join(args.outputs_dir, 'best_{}.pth'.format(tool_name[idx])))
            print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
