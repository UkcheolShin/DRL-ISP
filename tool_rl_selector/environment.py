import numpy as np
import torch

import sys # load img modifer
sys.path.append("../")
from common.CNN_tool_v2 import CNN_TOOL_V2
from common.trad_tool import TRA_TOOLS
import functools

import os
import time
TDEBUG = False

## DRL-ISP Environment
# @brief RL Environment
# @param config
# @return None
class MyEnvironment(object):
    def __init__(self, config):
        if 'file_path' in vars(config):
            config = config.config
        self.device = config.device

        # load task-specific dataload & reward generator
        self.app    = config.app
        if self.app == 'detection' :
            from reward.object_detection import Data_ObjDetection, Rewarder_ObjDetection
            self.dataloader     = Data_ObjDetection(config)
            map_calc = self.dataloader.map_calc
            self.rewarder       = Rewarder_ObjDetection(config, map_calc)
        elif self.app == 'depth' :
            from reward.depth_estimation import Data_DepthEstimation, Rewarder_DepthEstimation
            self.dataloader     = Data_DepthEstimation(config)
            self.rewarder       = Rewarder_DepthEstimation(config)
        elif self.app == 'restoration' :
            from reward.image_restoration import Data_ImgRestoration, Rewarder_ImgRestoration
            self.dataloader     = Data_ImgRestoration(config)
            self.rewarder       = Rewarder_ImgRestoration(config)

        self.use_factor = config.use_tool_factor
        _, self.height, self.width = self.dataloader.get_data_shape()
        self.prev_output_imgs = []
        self.prev_err_imgs = []

        # enviromental parameter for training
        self.train_data = None
        self.train_data_gt = None

        # enviromental parameter for testing
        self.test_data = None
        self.test_data_gt = None

        # shared enviromental parameter for training / testing
        self.episode_step = 0                              # current episode_step restoration step
        self.max_episode_steps = config.episode_max_step   # maximum episode step constraint
        self.done_basic = self._done_function_basic

        # build toolbox
        if config.not_use_cnn_tools:
            tool_cnn_list = []
            tool_layers = []
        else:
            tool_cnn_list = ['CTCNet', 'DeBlurNet', 'DeJpegNet', 'DeNoiseNet', 'ExposureNet', 'SRNet', 'WBNet']
            tool_layers = [[3, 8], [3, 8], [3, 8], [3, 8], [3, 8], [3, 8], [3, 8]]

        self.tool_name = []
        self.tools = []
        continuous_len = len(tool_cnn_list)

        # Add CNN-based tools
        for idx in range(len(tool_cnn_list)):
            for layer in tool_layers[idx]:
                network_name = f'CNN_TOOL{layer}'
                Net_Tool = CNN_TOOL_V2(num_channels=4, converter = None, use_skips=True, network=network_name).to(self.device)
                checkpoint = torch.load(os.path.join(config.toolpath, f'{tool_cnn_list[idx]}_L{layer}.pth'))
                Net_Tool.load_state_dict(checkpoint)
                for param in Net_Tool.parameters():
                    param.required_grad = False
                Net_Tool.eval()
                self.tools.append(Net_Tool)
                self.tool_name.append(f'{tool_cnn_list[idx]}_L{layer}')

        # Add Traditional-based tools
        # print('tra tools removed')
        trad_tool_list = []
        if not config.not_use_trad_tools:
            for x,y in TRA_TOOLS.__dict__.items() :
                if (type(y) == type(lambda:0)) and not x.startswith('_'):# from types import FunctionType, lambda:0 ->FunctionType :
                    trad_tool_list.append(x)

            TT = TRA_TOOLS()
            for idx, fcn in enumerate(trad_tool_list) :
                if self.use_factor and len(TT.config[fcn].default_factors) > 1:
                    for idx in range(len(TT.config[fcn].default_factors)):
                        factor = TT.config[fcn].default_factors[idx]
                        new_func = functools.partial(getattr(TT, fcn), factor=factor)
                        self.tools.append(new_func)
                        self.tool_name.append('{}_{:.2f}'.format(fcn, factor))
                else:
                    self.tools.append(getattr(TT, fcn))
                    self.tool_name.append(fcn)

        self.tool_name.append('done')
        self.action_size = len(self.tools) + 1

    ## DRL-ISP train image gatter function
    # @brief RL reset() function.
    #        read single image and gt file from h5py data
    #        initialize environmental params
    # @param None
    # @return self.train_data return single image
    def get_train_img(self):          # initialize environmental param
        self.episode_step = 0
        train_data, train_data_gt = self.dataloader.get_train_data()
        self.train_data    = train_data.to(self.device)
        if self.app == 'detection' :
            self.train_data_gt = train_data_gt
        else:
            self.train_data_gt = train_data_gt.to(self.device)
        self.rewarder.reset_train(self.train_data, self.train_data_gt)
        return self.train_data


    ## DRL-ISP step function
    # @brief RL step() function
    #        apply action to the current state
    #        and return next_state, reward, done
    # @param action
    # @return next_state, reward, done
    def step(self, action):
        if TDEBUG :
            start = time.time()

        action_done = reward_done = episode_done = False
        reward = 0.0

        if action == self.action_size - 1: # stop action, hum..
            reward = 0.
            action_done = True
        else: # Do action
            img_out = self.tools[action](self.train_data)
            self.train_data = img_out.clamp(0.0, 1.0)
            episode_done = self.done_basic()
            reward, reward_done = self.rewarder.get_reward_train(self.train_data, self.train_data_gt)

        self.episode_step += 1

        if TDEBUG :
            end = time.time()
            print('[one training] app : %s, excution time: %.4f' % (self.app, end-start))
        done = action_done or reward_done or episode_done
        not_include_replay = episode_done  # or reward_done or action_done

        if not_include_replay:
            return None, reward, done # next state, reward, done,
        else :
            return self.train_data, reward, done # next state, reward, done,

    ## DRL-ISP test image gatter function
    # @brief RL reset() function.
    #        read batchsize image and gt files from the target folder
    #        initialize environmental params
    # @param batch_size
    # @return self.test_data return batchsize images
    # @return test_done return done flag for current testset
    @torch.no_grad()
    def get_test_imgs(self, batch_size):
        self.episode_step = 0
        if self.app == 'detection':
            test_data, test_data_gt, test_done, idx_range = self.dataloader.get_test_data()
        else:
            test_data, test_data_gt, test_done = self.dataloader.get_test_data()
            idx_range = None
        self.test_data = test_data.to(self.device)

        if self.app == 'detection' :
            self.test_data_gt = test_data_gt
            img, err_img = self.rewarder.reset_test(self.test_data, self.test_data_gt, return_img=True, idx_range=idx_range)
        else:
            self.test_data_gt = test_data_gt.to(self.device)
            img, err_img = self.rewarder.reset_test(self.test_data, self.test_data_gt, return_img=True)
        self.prev_output_imgs = list(img.split(1))
        self.prev_err_imgs = list(err_img.split(1))
        return self.test_data, test_done, self.rewarder.get_playscore(), img, err_img

    ## DRL-ISP test step function
    # @brief RL step() function with batchsize
    #        apply action to the current state
    #        and return next_state, reward, done
    # @param action
    # @return next_state, reward, done
    @torch.no_grad()
    def step_test(self, actions):
        rewards = np.zeros(actions.shape)
        dones = np.zeros(actions.shape, dtype=bool) # all False matrix
        action_dones = np.zeros(actions.shape, dtype=bool)
        reward_dones = np.zeros(actions.shape, dtype=bool)
        episode_dones = np.zeros(actions.shape, dtype=bool)
        if TDEBUG :
            start = time.time()
        # loop actions because the actions are all different
        output_imgs = []
        err_imgs = []
        for k, action in zip(range(len(actions)), actions) :
            # if the action == stop, preseve current setup.
            if action == self.action_size - 1: # stop action
                #self.test_data[k] = None # unchanged
                rewards[k] = 0.
                action_dones[k] = True
                output_imgs.append(self.prev_output_imgs[k])
                err_imgs.append(self.prev_err_imgs[k])
            else:  # else, do action
                try:
                    img_out = self.tools[action](self.test_data[[k]])
                except:
                    print('hi')
                self.test_data[[k]] = img_out.clamp(0.0, 1.0)
                episode_dones[k] = self.done_basic()
                if self.app == 'detection' :
                    rewards[k], reward_done, output_img, err_img = self.rewarder.get_reward_test(self.test_data[[k]], self.test_data_gt[k], k, return_img=True, img_step=self.episode_step + 1)
                    # output_img = img_out
                    # err_img = img_out
                else:
                    rewards[k], reward_done, output_img, err_img = self.rewarder.get_reward_test(self.test_data[[k]], self.test_data_gt[[k]], k, return_img=True)
                output_imgs.append(output_img)
                err_imgs.append(err_img)
                reward_dones[k] = reward_done
                self.prev_output_imgs[k] = output_img
                self.prev_err_imgs[k] = err_img

        self.episode_step += 1

        dones = np.logical_or(action_dones, episode_dones)  # exclude psnr cut in test
        if TDEBUG :
            end = time.time()
            print('[batch testing] app : %s, excution time: %.4f' % (self.app, end-start))
        return self.test_data, rewards, dones, self.rewarder.get_playscore(), torch.cat(output_imgs, dim=0), torch.cat(err_imgs, dim=0)


    ## DRL-ISP inplace done function
    # @brief decide done flag according to episode_step
    def _done_function_basic(self):
        if self.episode_step >= self.max_episode_steps - 1: # step constraint
            return True
        return False