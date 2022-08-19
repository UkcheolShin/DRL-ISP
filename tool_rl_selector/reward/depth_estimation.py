import torch
import numpy as np
import cv2
import os
import h5py
import math
import random

# equired lib for depth
import torch.utils.data as data
from imageio import imread 
from path import Path
from reward import custom_transforms
from reward.depth import models
from skimage.transform import resize as imresize
from reward.utils import ETH_img_load, syn_img_load, load_as_float, raw4ch, bayer_to_rgb

def crawl_folders(folders_list, dataset='kitti'):
    imgs = []
    depths = []
    type1 = None
    for folder in folders_list:
        if type1 is None:
            files = folder.files()
            for idx in range(len(files)):
                if files[idx][-4:] in [".jpg", ".png"]:
                    type1 = '*' + folder.files()[idx][-4:]
                    break

        current_imgs = sorted(folder.files(type1))

        # remove unvalid files
        for name, idx in zip(current_imgs, range(len(current_imgs))) : 
            if 'temp' in name:
                current_imgs.pop(idx)

        if dataset == 'kitti':
            current_depth = sorted(folder.files('*.npy'))
        imgs.extend(current_imgs)
        depths.extend(current_depth)

    if len(imgs) != len(depths):
        print('Err in files. Are there .jpg_temp.png files?')
        raise NameError
    return imgs, depths


## DRL-ISP Dataloader
# @brief RL Dataloader 
# @param config
# @return None
class Data_DepthEstimation(object):
    def __init__(self, config):
        self.is_train = ~config.is_test

        self.root = Path(config.train_dir)
        transforms = [custom_transforms.ArrayToTensor()]
        self.transform = custom_transforms.Compose(transforms)
        self.dataset = 'kitti'
        self.original_train_len = 0
        self.used_train_len = 0
        self.original_test_len = 0
        self.used_test_len = 0
        # 1. Training dataset
        if self.is_train:
            # 1-1. Read training data
            scene_list_path = self.root/'train.txt' 
            self.train_list = [self.root/folder[:-1] for folder in open(scene_list_path)]
            self.data, self.label = crawl_folders(self.train_list, self.dataset)

            # shuffle order
            random.seed(4)
            random.shuffle(self.data)
            random.seed(4)
            random.shuffle(self.label)
            self.original_train_len = len(self.data)
            self.used_train_len = len(self.data)
            if config.train_dataset_len != 0:
                train_len = config.train_dataset_len
                self.used_train_len = train_len
                self.data      = self.data[:train_len]
                self.label     = self.label[:train_len]

            # 1-2. set data index
            self.train_data_index = 0
            self.train_data_len = len(self.data)

            _, self.train_height, self.train_width = imread(self.data[0]).transpose(2,0,1).shape
        else:
            raise ValueError('Invalid dataset!')

        # 2. Testing dataset
        # 2-1. Read testing data
        test_img_folder = os.path.join(config.test_dir, 'color')
        test_label_folder = os.path.join(config.test_dir, 'depth')
        self.list_in, self.list_gt = crawl_folders([Path(test_img_folder), Path(test_label_folder)])
        # shuffle order
        random.seed(4)
        random.shuffle(self.list_in)
        random.seed(4)
        random.shuffle(self.list_gt)
        self.original_test_len = len(self.list_in)
        self.used_test_len = len(self.list_in)

        if config.test_dataset_len >0:
            test_len = config.test_dataset_len
            self.used_test_len = test_len
            self.list_in  = self.list_in[:test_len]
            self.list_gt = self.list_gt[:test_len]

        # current data index
        self.test_data_index = 0
        self.test_data_len = len(self.list_in)

        _, self.test_height, self.test_width = imread(self.list_in[0]).transpose(2,0,1).shape
        self.test_resize = False
        if self.test_height != self.train_height or self.test_width != self.train_width:
            self.test_resize = True
        self.last_test_name = ''

    # output shape : 1CHW
    def get_train_data(self) : 
        # if data index exceed the dataset range, load new dataset
        if self.train_data_index >= self.train_data_len:
            self.train_data_index = 0

        img = imread(self.data[self.train_data_index])
        img = img.astype(np.float32)
        img = raw4ch(img)
        img = self.transform([img])
        train_data = img[0]
        train_data_gt = torch.from_numpy(np.load(self.label[self.train_data_index]).astype(np.float32))

        self.train_data_index += 1
        return train_data.unsqueeze(0), train_data_gt.unsqueeze(0)

    def get_test_data(self, batch_size=1):
        # read images from testset folder with batch number
        idx_start = self.test_data_index
        idx_end  = idx_start + batch_size
        if idx_end >= self.test_data_len:
            idx_end = self.test_data_len
            self.test_data_index = 0
            test_done = True
        else :
            self.test_data_index += batch_size
            test_done = False

        list_in  = self.list_in[idx_start:idx_end] 
        list_gt  = self.list_gt[idx_start:idx_end] 

        img_num   = len(list_in)
        test_imgs = []
        test_gts = []

        for k in range(img_num):
            img = imread(list_in[k])
            img = img.astype(np.float32)
            img = raw4ch(img)
            if self.test_resize:
                img = imresize(img, (self.train_height, self.train_width))

            test_imgs.append((self.transform([img])[0]).unsqueeze(0))
            test_gt = np.load(list_gt[k]).astype(np.float32)
            test_gts.append(torch.from_numpy(test_gt).unsqueeze(0))
        test_imgs = torch.cat(test_imgs, dim=0)
        test_gts = torch.cat(test_gts, dim=0)
        return test_imgs, test_gts, test_done

    def get_data_shape(self):
        return imread(self.data[0]).transpose(2,0,1).shape


############################################################
## DRL-ISP Rewarder
# @brief RL Reward Generator 
# @param config
# @return None
class Rewarder_DepthEstimation(object):
    def __init__(self, config):
        self.device = config.device
        # model load    
        self.disp_net = models.DispResNet(18, False).to(self.device)
        weights = torch.load('reward/depth/weight/dispnet_model_best.pth.tar')
        self.disp_net.load_state_dict(weights['state_dict'])
        self.disp_net.eval()
        for param in self.disp_net.parameters():
            param.required_grad = False

        # enviromental parameter for training
        self.done_function = self._done_function
        self.metric_init = 0.
        self.metric_prev = 0.
        
        # enviromental parameter for testing
        self.done_function_test = self._done_function_with_idx
        self.metric_init_test = 0.
        self.metric_prev_test = 0.

        # reward functions
        self.metric_function = self._sup_error_cal
        self.reward_function = self._step_metric_reward
        self.get_playscore   = self.get_step_test_metric
        self.scaler = config.reward_scaler
        self.reward_func_name = config.reward_func

    def normalize_image(self, img):
        return ((img-0.45)/0.225)

    ## DRL-ISP train param initializer
    # @brief 
    @torch.no_grad()
    def reset_train(self, train_data, train_data_gt) : 
        # initialize environmental param
        metric_init         = self.metric_function(train_data, train_data_gt)
        self.metric_init    = metric_init
        self.metric_prev    = metric_init
        return None

    ## DRL-ISP test param initializer
    # @brief 
    @torch.no_grad()
    def reset_test(self, test_data, test_data_gt, return_img=False) :
        # initialize environmental param
        if self.reward_func_name == 'all':
            self.metric_init_test = torch.zeros(size=(len(test_data), 8))
            self.metric_prev_test = torch.zeros(size=(len(test_data), 8))
        else:
            self.metric_init_test = torch.zeros(len(test_data))
            self.metric_prev_test = torch.zeros(len(test_data))

        imgs = []
        err_imgs = []
        for k in range(len(test_data)):
            if return_img:
                init_metric, output_img, err_img = self.metric_function(test_data[[k], ...], test_data_gt[[k], ...], return_img=True)
                imgs.append(output_img)
                err_imgs.append(err_img)
            else:
                init_metric = self.metric_function(test_data[[k], ...], test_data_gt[[k], ...], return_im=False)
            self.metric_init_test[k] = init_metric
            self.metric_prev_test[k] = init_metric
        if return_img:
            return torch.cat(imgs, dim=0), torch.cat(err_imgs, dim=0)
        return None

    ## DRL-ISP Img restoration reward getter (train)
    # @brief 
    @torch.no_grad()
    def get_reward_train(self, data_in, data_gt):
        metric_cur = self.metric_function(data_in, data_gt)
        reward = self.reward_function(metric_cur)
        done = self.done_function(metric_cur)
        self.metric_prev = metric_cur
        return reward, done

    ## DRL-ISP Img restoration reward getter (test)
    # @brief 
    @torch.no_grad()
    def get_reward_test(self, data_in, data_gt, idx, return_img=False):
        if return_img:
            metric_cur, output_img, err_img = self.metric_function(data_in, data_gt, return_img=True)
        else:
            metric_cur = self.metric_function(data_in, data_gt, return_img=False)
        reward = self.reward_function(metric_cur, idx)
        done = self.done_function_test(metric_cur, idx)
        self.metric_prev_test[idx] = metric_cur
        if return_img:
            return reward, done, output_img, err_img
        return reward, done

    ## DRL-ISP Img restoration metric getter (test)
    # @brief 
    @torch.no_grad()
    def get_step_test_metric(self): 
        return self.metric_prev_test

    ## DRL-ISP inplace metric calculation function
    # @brief 
    @torch.no_grad()
    def _sup_error_cal(self, data_in, data_gt, return_img=False):
        _, C_, _, _ = data_in.shape
        if C_ == 4:
            data_in = bayer_to_rgb(data_in)

        output_disp = self.disp_net(self.normalize_image(data_in))
        output_depth = 1/output_disp.squeeze(1)

        if data_gt.nelement() != output_depth.nelement():
            b, h, w = data_gt.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)
        # [abs_diff, abs_rel, sq_rel, a1, a2, a3, rmse, rmslog]
        if return_img:
            metrics, imgs, factors = self.compute_errors(data_gt, output_depth, return_img=True)
        else:
            metrics, factors = self.compute_errors(data_gt, output_depth, return_img=False)
        if self.reward_func_name == 'rmse':
            metric_cur = metrics[6]
        elif self.reward_func_name == 'delta1':
            metric_cur = metrics[3]
        elif self.reward_func_name == 'delta2':
            metric_cur = metrics[4]
        elif self.reward_func_name == 'delta3':
            metric_cur = metrics[5]
        elif self.reward_func_name == 'absrel':
            metric_cur = metrics[1]
        elif self.reward_func_name == 'sqrel':
            metric_cur = metrics[2]
        elif self.reward_func_name == 'absdiff':
            metric_cur = metrics[0]
        elif self.reward_func_name == 'rmslog':
            metric_cur = metrics[7]
        elif self.reward_func_name == 'all':
            metric_cur = torch.FloatTensor(metrics)
        else:
            raise NotImplementedError

        if return_img:
            return metric_cur, output_depth * factors[0], imgs
        return metric_cur

    ## DRL-ISP inplace reward function
    # @brief decide done flag according to metric difference, episode_step
    @torch.no_grad()
    def _step_metric_reward(self, metric_cur, idx=None):
        if self.reward_func_name == 'all':
            if idx == None :
                reward = self.metric_prev[6] - metric_cur[6] # if RMSE error, cur error is lower than prev, i.e, prev_err > cur_err
            else:
                reward = self.metric_prev_test[idx][6] - metric_cur[6]  # if RMSE error, cur error is lower than prev, i.e, prev_err > cur_err
        elif self.reward_func_name in ['rmse', 'absrel', 'sqrel', 'absdiff', 'rmslog']:
            if idx == None :
                reward = self.metric_prev - metric_cur # if RMSE error, cur error is lower than prev, i.e, prev_err > cur_err
            else:
                reward = self.metric_prev_test[idx] - metric_cur # if RMSE error, cur error is lower than prev, i.e, prev_err > cur_err
        elif self.reward_func_name in ['delta1', 'delta2', 'delta3']:
            if idx == None :
                reward = metric_cur - self.metric_prev  # other metrics are reversed
            else:
                reward = metric_cur - self.metric_prev_test[idx]
        else:
            raise NotImplementedError
        return reward * self.scaler


    ## DRL-ISP inplace done function
    # @brief decide done flag according to metric difference, episode_step
    @torch.no_grad()
    def _done_function(self, metric_cur):
        if self.reward_func_name == 'all':
            if self.metric_init[6] - metric_cur[6] < -5:  # RMSE is too bad, stop.
                return True
        elif self.metric_init - metric_cur < -5 :  # RMSE is too bad, stop.
            return True
        return False

    ## DRL-ISP inplace done function
    # @brief decide done flag according to metric difference, episode_step
    def _done_function_with_idx(self, metric_cur, idx):
        if self.reward_func_name == 'all':
            if self.metric_init_test[idx][6] - metric_cur[6] < -5 :  # metric is too bad
                return True
        elif self.metric_init_test[idx] - metric_cur < -5:  # metric is too bad
            return True

        return False

    ## DRL-ISP metric calculation function
    # @brief 
    @torch.no_grad()
    def compute_errors(self, gt, pred, return_img=False):
        abs_diff, abs_rel, sq_rel, a1, a2, a3, rmse, rmslog = 0, 0, 0, 0, 0, 0, 0, 0
        batch_size, h, w = gt.size()

        '''
        crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        construct a mask of False values, with the same size as target
        and then set to True values inside the crop
        '''
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80
        err_imgs = []
        factors = []
        for current_gt, current_pred in zip(gt, pred):
            valid = (current_gt > 0.1) & (current_gt < max_depth)
            valid = valid & crop_mask

            valid_gt = current_gt[valid]
            valid_pred = current_pred[valid].clamp(1e-3, max_depth)
            factor = torch.median(valid_gt)/torch.median(valid_pred)
            valid_pred = valid_pred * factor

            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).float().mean()
            a2 += (thresh < 1.25 ** 2).float().mean()
            a3 += (thresh < 1.25 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
            rmse += (valid_gt - valid_pred) ** 2
            rmslog += (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
            factors.append(factor / max_depth)
            if return_img:
                pred_img = current_pred * valid.float() * factor
                gt_img = current_gt * valid.float()
                err_imgs.append(torch.abs(gt_img - pred_img).unsqueeze(0) / max_depth)

        rmse = torch.sqrt(torch.mean(rmse))
        rmslog = torch.sqrt(torch.mean(rmslog))

        if return_img:
            return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]] + [rmse, rmslog], torch.cat(err_imgs, dim=0), factors
        return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]] + [rmse, rmslog], factors


import argparse
from torch.autograd import Variable
import numpy as np

# debug & test
parser = argparse.ArgumentParser(description='pytorch implementation for RL-restore',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# network parameter 
parser.add_argument('--is_test', action='store_true', help='Whether to do training or testing')
parser.add_argument('--app', type=str, choices=['detection', 'segmentation', 'depth', 'restoration'], default='restoration', help='the dataset to train')

parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='n')

parser.add_argument('--save_dir', type=str, default='./results/', help='Path for saving models and playing result')
parser.add_argument('--train_dir', type=str, default='dataset/train/', help='Path for training model')
parser.add_argument('--val_dir', type=str, default='dataset/valid/', help='Path for validating model')
parser.add_argument('--test_dir', type=str, default='dataset/test/', help='Path for testing model')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__=='__main__':
    args = parser.parse_args()
    args.device = device
    args.train_dir = '/DB/Dataset_DRLISP/KITTI_depth/KITTI_sc'

    # check data loader
    dataloader     = Data_DepthEstimation(args)
    train_img, train_gt = dataloader.get_train_data()

    print('train img shape :' + str(train_img.shape)); print('train gt shape :' + str(train_gt.shape))
    test_imgs, test_gts, done = dataloader.get_test_data(32)
    print('test img shape :' + str(test_imgs.shape)); print('test gt shape :' + str(test_gts.shape));  print('test done :' + str(done))
    print('dataloader.get_dat_shape :' + str(dataloader.get_data_shape()))

    rewarder       = Rewarder_DepthEstimation(args)
    rewarder.reset_train(train_img.to(device), train_gt.to(device))
    rewarder.reset_test(test_imgs.to(device), test_gts.to(device))

    train_reward, done = rewarder.get_reward_train(train_img.to(device), train_gt.to(device))
    print('train_reward:' + str(train_reward)); print('train done:' + str(done)); 
    idx=1
    test_rewards, done = rewarder.get_reward_test(test_imgs[[idx]].to(device), test_gts[[idx]].to(device), idx)
    print('test_reward:' + str(test_rewards)); print('test done:' + str(done)); 
    cur_score = rewarder.get_step_test_metric()
    print('test batch cur reward :' + str(cur_score)); print('test idx reward:' + str(cur_score[idx])); 
    