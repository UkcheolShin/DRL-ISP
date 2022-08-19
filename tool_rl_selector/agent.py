import cv2
import numpy as np
import os
import random
import math
import copy
import time

from replay_memory import ReplayMemory, PriReplayMemory
from models.dqn import DQN
from models.TSnet import ToolSelectNet
from models.sac import SACPolicy, SACQNetwork
from logger import Logger, Printer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import datetime
import json
from utils import prAuto, disable_gradients, getBack
import shutil
import tqdm
from skimage.transform import resize as imresize
from reward.utils import bayer_to_rgb

import itertools
#import warnings
#warnings.filterwarnings("ignore")

## DRL-ISP Environment
# @brief RL Environment
# @param config
# @return None
# This class play agent to select a action according to the current state.
class Agent(object):
    def __init__(self, config, environment, default_config):
        super(Agent, self).__init__()
        self.config = config
        self.device = config.device
        self.cnn_format = 'BCHW'
        self.is_continue = (config.continue_file != '')

        # agent param
        self.model = config.agent
        self.env = environment
        if config.per : 
            self.memory = PriReplayMemory(config.memory_size)
        else:
            self.memory = ReplayMemory(config.memory_size)

        self.action_size = self.env.action_size

        # training param
        self.step = 0
        self.stop_step = config.episode_max_step
        self.target_update_step = config.target_q_update_step          # update period of target network
        self.train_episodes_num = config.train_episodes        # total training episode number
        self.test_step = 3000
        self.cur_episode = 0

        # testing param
        self.play_episode_interval = 5000      # test dataset playing period
        self.play_count = 0
        self.save_ratio = 1.0
        self.play_batch = config.play_batch
        
        # action selection param
        self.eps_start = 1.0                # ep_start = 1.  # 1: fully random; 0: no random
        self.eps_end = 0.05                 # ep_end = 0.1
        self.eps_decay = 50000 

        # optimize param
        self.batch_size = config.batch_size
        self.clip: bool = True        
        self.gamma = 0.99
        self.reward_clip = config.reward_clip

        # save param
        self.is_save = config.is_save
        self.save_folder = config.save_dir
        if self.save_folder != '' and self.is_save:
            os.makedirs(self.save_folder, exist_ok=True)
            with open(os.path.join(self.save_folder, 'config.json'), 'w') as f:
                config_dict = copy.deepcopy(vars(config))
                config_dict['device'] = str(config_dict['device'])
                f.write(json.dumps(config_dict, indent=2))

        if config.clear_previous_result:
            flag = False
            while not flag:
                value = input(prAuto('[WARNING] Attempt to clear previous results. type (yes/no): '))
                if value.lower() == 'yes':
                    if os.path.exists(self.save_folder):
                        shutil.rmtree(self.save_folder)
                    os.makedirs(self.save_folder, exist_ok=True)
                    flag = True
                elif value.lower() == 'no':
                    flag = True
                else:
                    print('Invalid input')
        self.best_reward = -1
        self.best_score  = -1
        self.best_psnr   = 0

        # SAC
        self.start_timestep = config.start_timestep
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.target_entropy_ratio = config.target_entropy_ratio
        self.feature_extractor = config.feature_extractor
        self.train_frequency = config.policy_freq
        
        # DQN Model
        self.dqn_models = ['dqn_cnn', 'dqn_hist']
        self.feature_replay = False

        if self.model == 'dqn_cnn' :
            self.policy_net = DQN(self.env.height, self.env.width, self.action_size).to(self.device)
            self.target_net = DQN(self.env.height, self.env.width, self.action_size).to(self.device)
            self.target_net.eval()
        elif self.model == 'dqn_hist' :
            self.policy_net = ToolSelectNet(pretrained=True, num_action=self.action_size).to(self.device)
            self.target_net = ToolSelectNet(pretrained=True, num_action=self.action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        elif self.model == 'sac':
            if self.feature_extractor == 'hist' or self.feature_extractor == 'rand':
                self.policy_net = SACPolicy(self.env.height, self.env.width, self.action_size,
                                            feature_extractor=self.feature_extractor, resize_denominator=config.resize_denominator,
                                            layers=config.layers, layer_hidden=config.layer_hidden,
                                            layer_activation=config.layer_activation,
                                            not_use_inten=config.not_use_inten,
                                            not_use_grad=config.not_use_grad,
                                            not_use_seman=config.not_use_seman,
                                            seman_net_name=config.seman_net_name
                                            ).to(self.device)
                feature_size = self.policy_net.feature_size
                self.online_critic = SACQNetwork(feature_size, None, self.action_size,
                                                 feature_extractor='shared', layers=config.layers, layer_hidden=config.layer_hidden,
                                                 layer_activation=config.layer_activation).to(self.device)
                self.target_critic = SACQNetwork(feature_size, None, self.action_size, feature_extractor='shared',
                                                 layers=config.layers, layer_hidden=config.layer_hidden, layer_activation=config.layer_activation).to(self.device)
                self.feature_replay = True
            else:
                self.policy_net = SACPolicy(self.env.height, self.env.width, self.action_size,
                                            feature_extractor=self.feature_extractor, resize_denominator=config.resize_denominator,
                                            layers=config.layers, layer_hidden=config.layer_hidden, layer_activation=config.layer_activation,
                                            seman_net_name=config.seman_net_name).to(self.device)
                self.online_critic = SACQNetwork(self.env.height, self.env.width, self.action_size,
                                                 feature_extractor=self.feature_extractor, resize_denominator=config.resize_denominator,
                                                 layers=config.layers, layer_hidden=config.layer_hidden,
                                                 layer_activation=config.layer_activation).to(self.device)
                self.target_critic = SACQNetwork(self.env.height, self.env.width, self.action_size,
                                                 feature_extractor=self.feature_extractor, resize_denominator=config.resize_denominator,
                                                 layers=config.layers, layer_hidden=config.layer_hidden,
                                                 layer_activation=config.layer_activation).to(self.device)
            self.alpha = config.sac_alpha

            self.target_critic.load_state_dict(self.online_critic.state_dict())
            self.target_critic.eval()

            disable_gradients(self.target_critic)

        # optimize
        if self.model == 'sac':
            if self.feature_extractor == 'cnn':
                self.q_optim = optim.Adam(self.online_critic.get_parameters(), lr=0.0001)
                self.policy_optim = optim.Adam(
                    list(self.policy_net.feature.parameters()) +
                    list(self.policy_net.head.parameters()), lr=0.0001)

            elif self.feature_extractor == 'hist' or self.feature_extractor == 'rand':
                self.q1_optim = optim.Adam(self.online_critic.q1.parameters(), lr=0.0001)
                self.q2_optim = optim.Adam(self.online_critic.q2.parameters(), lr=0.0001)
                self.policy_optim = optim.Adam(self.policy_net.head.parameters(), lr=0.0001)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)

        if self.is_continue:
            self.load_checkpoint(config.continue_file)
            if config.cur_episode != -1:
                self.cur_episode = config.cur_episode
            if config.play_count != -1:
                self.play_count = config.play_count

        cur_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        config_dict = vars(config)

        def make_log_folder_name():
            dqn_keys = ['agent', 'app']
            sac_keys = ['agent', 'feature_extractor', 'app', 'resize_denominator']
            exclude_keys = ['train_dir', 'val_dir', 'test_dir', 'clear_previous_result', 'log_dir',
                            'is_save', 'cur_episode', 'continue_file', 'play_count']
            filename_prohibit = ['/', ]
            if self.model in self.dqn_models:
                keys = dqn_keys
            elif self.model == 'sac':
                keys = sac_keys
            else:
                raise NotImplementedError

            str1 = ''
            for idx, key in enumerate(keys):
                str1 += '{}={}.'.format(key, config_dict[key])

            for key in default_config:
                if config_dict[key] != default_config[key]:
                    if key not in keys and key not in exclude_keys:
                        str2 = str(config_dict[key])
                        for prohibit in filename_prohibit:
                            str2 = str2.replace(prohibit, '_')

                        str1 += '{}={}.'.format(key, str2)
            if len(str1) > 230:
                str1 = str1[:230] + '...'
            str1 += cur_time
            return str1

        self.log_folder = os.path.join(config.log_dir, make_log_folder_name())
        os.makedirs(self.log_folder, exist_ok=True)

        config_dict['device'] = str(config_dict['device'])
        self.logger = Logger(self.log_folder)
        self.logger.log('== Default Values ==', logonly=True)
        self.logger.log(json.dumps(default_config, indent=4), logonly=True)
        self.logger.log('== Current Values ==', logonly=True)
        self.logger.log(json.dumps(config_dict, indent=4))
        self.logger.log(f'Total number of tools: {self.action_size}')
        for idx, name in enumerate(self.env.tool_name):
            self.logger.log(f'{idx:02d}:  {name}')
        self.graph_write = False

    ## Select action according to exploration policy 
    # @brief Select action according to exploration policy 
    @torch.no_grad()
    def select_action(self, state, prev_action, test_ep=None) :
        if self.model == 'sac':
            feature1 = None

            if self.step < self.start_timestep: # random exploration
                logit = (torch.randn((state.shape[0], self.action_size)) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                action_vec_ = F.softmax(logit, dim=1)
                action = action_vec_.max(1)[1] if test_ep else action_vec_.multinomial(num_samples=1).data[0]
                if self.feature_replay:
                    feature1 = self.policy_net.forward_feature(state)
                return action, action_vec_, feature1
            else: 
                if test_ep is not None:
                    feature1, action_logits, greedy_actions = self.policy_net.select_action(state)
                    # action_vec_ = F.softmax(action_logits, dim=1)
                    return greedy_actions.squeeze(1), action_logits, feature1
                else:
                    feature1, actions, action_probs, log_action_probs = self.policy_net(state)
                    # action_vec_ = F.softmax(action_probs, dim=1)
                    return actions.squeeze(1), action_probs, feature1  # [1], [1, 31], [1, 5376]
        else:
            # DQN version
            eps_threshold = test_ep or (self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * (self.step  - self.config.learn_start)/ self.eps_decay))

            # greedy search
            if random.random() > eps_threshold:
                action_vec_ = self.policy_net(state)
                action = action_vec_[0].argmax(axis=0)
                action = action.unsqueeze(0).unsqueeze(1)
                return action, action_vec_, None
            else:
                action = random.randrange(self.action_size)
                action_vec_ = torch.zeros([1, self.action_size]).to(self.device)
                action_vec_[0, action] = 1.0
                return torch.tensor([[action]], dtype=torch.long).to(self.device), action_vec_, None

    ## Train the agent according to batch of replay buffer
    # @brief optimize model with randomly sampled batch
    # @param None
    # @return None
    # @todo update RL update method
    def optimize_model(self):
        # initialization
        self.policy_net.train()

        if len(self.memory) < self.batch_size:
            return

        # get samples from replay buffer
        if self.config.per :
            transitions, indices, weights = self.memory.sample(self.batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)

        # get mask
        non_final_mask = torch.BoolTensor(list(map(lambda ns: ns is not None, transitions.next_state))).to(self.device)
        final_mask     = ~non_final_mask

        state_batch  = torch.cat(transitions.state).float().to(self.device)
        action_batch = torch.cat(transitions.action).float().to(self.device)
        reward_batch = torch.cat(transitions.reward).float().to(self.device)
        action_one_batch = torch.cat(transitions.action_one).long().to(self.device)
        mean_reward = float(torch.sum(reward_batch).data.cpu().numpy())
        try : 
            non_final_next_state_batch = torch.cat([ns for ns in transitions.next_state if ns is not None]).float().to(self.device)
        except:
            return 0, mean_reward, 0, 0

        # Reshape States and Next States, B3WH
        if not self.feature_replay:
            state_batch                = state_batch.view([self.batch_size, 3, self.env.width, self.env.height])
            non_final_next_state_batch = non_final_next_state_batch.view([-1, 3, self.env.width, self.env.height])

        # Clipping Reward between -2 and 2
#        reward_batch.data.clamp_(-self.reward_clip, self.reward_clip)

        # Predict q value of time t by DQN Model
        if self.model in self.dqn_models :
            q_pred = self.policy_net(state_batch)
            q_values = q_pred.gather(1, action_batch.argmax(1).unsqueeze(1))
        elif self.model == 'sac':
            if self.feature_extractor == 'hist' or self.feature_extractor == 'rand':
                q_pred1 = self.online_critic.q1(state_batch)
                q_pred2 = self.online_critic.q2(state_batch)  
                # q_pred1, q_pred2 = self.online_critic(policy_feature)
            else:
                q_pred1, q_pred2 = self.online_critic(state_batch)
            q_pred_action1 = q_pred1.gather(1, action_one_batch.unsqueeze(1)).squeeze(1)
            q_pred_action2 = q_pred2.gather(1, action_one_batch.unsqueeze(1)).squeeze(1)
            q_pred = q_pred1
        else:
            raise NotImplementedError

        # Predict by Target Model
        if self.model in self.dqn_models :
            with torch.no_grad():
                next_q_target_values = self.target_net(non_final_next_state_batch).max(1)[0].detach()
        elif self.model == 'sac':
                next_state = torch.cat([cs if ns is None else ns for cs, ns in zip(transitions.state, transitions.next_state)]).float().to(self.device)
                if self.feature_extractor == 'hist' or self.feature_extractor == 'rand':
                    with torch.no_grad():
                        next_q1, next_q2 = self.target_critic(next_state)
                        _, next_actions, next_action_probs, next_log_action_probs = self.policy_net.forward_without_feature(next_state)
                else:
                    with torch.no_grad():
                        next_q1, next_q2 = self.target_critic(next_state)
                        _, next_actions, next_action_probs, next_log_action_probs = self.policy_net(next_state)
                with torch.no_grad():
                    next_q_target_values = (next_action_probs * (torch.min(next_q1, next_q2) - self.alpha * next_log_action_probs)).sum(dim=1)
                next_q_target_values = next_q_target_values[non_final_mask]
        else:
            raise NotImplementedError

        with torch.no_grad():
            target_q_values                 = torch.zeros(self.batch_size).float().to(self.device)
            target_q_values[non_final_mask] = reward_batch[non_final_mask] + next_q_target_values * self.gamma
            target_q_values[final_mask]     = reward_batch[final_mask]

        actor_loss_float = 0.0
        if self.model in self.dqn_models:
            # Huber loss 
            if self.config.per :
                loss = (q_values - target_q_values.unsqueeze(1)) * weights
                prios = loss+1e-5
                loss = loss.mean()
                self.memory.update_priorities(indices, prios.data.cpu().numpy())
            else:
                loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

            # agent optimize
            self.optimizer.zero_grad()
            loss.backward()
        
            if self.clip:
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            loss_float = loss.detach().cpu().numpy()
        elif self.model == 'sac':
            # critic loss
            if self.config.per :
                q1_loss = (q_pred_action1 - target_q_values).pow(2)* torch.from_numpy(weights).cuda()
                q2_loss = (q_pred_action2 - target_q_values).pow(2)* torch.from_numpy(weights).cuda()
                prios = q1_loss + q2_loss+ 1e-5
                q1_loss = q1_loss.mean()
                q2_loss = q2_loss.mean()

                self.memory.update_priorities(indices, prios.data.cpu().numpy())
            else:
                q1_loss = torch.mean((q_pred_action1 - target_q_values).pow(2))
                q2_loss = torch.mean((q_pred_action2 - target_q_values).pow(2))

            # policy loss
            if self.feature_extractor == 'cnn':
                _, actions, action_probs, log_action_probs = self.policy_net(state_batch)
            else:
                _, actions, action_probs, log_action_probs = self.policy_net.forward_without_feature(state_batch)

            with torch.no_grad():
                q1, q2 = self.online_critic(state_batch)

            entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
            q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)
            policy_loss = ((-q - self.alpha * entropies)).mean()

            if self.feature_extractor == 'cnn':
                q_loss = q1_loss + q2_loss
                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()
            else:
                self.q1_optim.zero_grad()
                q1_loss.backward()
                self.q1_optim.step()

                self.q2_optim.zero_grad()
                q2_loss.backward()
                self.q2_optim.step()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            loss_float = q1_loss.detach().cpu().numpy()
            policy_loss_float = policy_loss.detach().cpu().numpy()
        else:
            raise NotImplementedError

        with torch.no_grad():
            reward_score = mean_reward
            q_mean = torch.sum(q_pred, 0).data.cpu().numpy()[0]
            target_mean = torch.sum(next_q_target_values, 0).data.cpu().numpy()

        if self.model == 'sac':
            return [loss_float, policy_loss_float], reward_score, q_mean, target_mean
        else:
            return loss_float, reward_score, q_mean, target_mean

    ## Train agent according to the training dataset
    # @brief Train agent
    # @param None
    # @return None
    def train(self):

        for i_episode in range(self.cur_episode, self.train_episodes_num+1):
            # Initialize environment and agent
            img_cur = self.env.get_train_img()  

            prev_action = torch.zeros(1, self.action_size).to(self.device)
            ep_reward = 0.

            if i_episode % 2000 == 0:
                self.logger.log('|    ', end='')
                for idx in range(self.action_size):
                    self.logger.log('|{:04d}'.format(idx), end='')
                self.logger.log('|')

            while True:
                # 1. agent predicts an action
                with torch.no_grad():
                    action, action_vec, feature_vec = self.select_action(img_cur, prev_action)
                # 2. environment conducts the action
                with torch.no_grad():
                    next_img, reward, done = self.env.step(action)

                # 3. store the information in Replay Memory
                img_cur_     = img_cur.detach().cpu()
                action_vec_  = action_vec.detach().cpu()
                if next_img is not None : 
                    next_img_    = next_img.detach().cpu()
                else:
                    next_img_    = None

                action_ = action.detach().cpu()

                replay_state = img_cur_
                replay_next_state = next_img_
                if self.feature_replay:
                    replay_state = feature_vec.detach().cpu()
                    if next_img is not None:
                        replay_next_state = self.policy_net.forward_feature(next_img).detach().cpu()

                if done :
                    if next_img == None :
                        self.memory.push(replay_state, action_vec_, 0, None, done, action_)  # action -1=255 for uint8
                    else:
                        self.memory.push(replay_state, action_vec_, reward, replay_next_state, False, action_)  # action -1=255 for uint8
                else:
                    self.memory.push(replay_state, action_vec_, reward, replay_next_state, done, action_)

                prev_action = action_vec[:, :-1]
                img_cur = next_img

                loss = None
                # 4. train agent network
                if self.memory.is_available(self.batch_size) and self.step > self.config.learn_start :
                    if self.step % self.train_frequency == 0:
                        loss, reward_batch, q_batch, traget_q_batch = self.optimize_model()

                ep_reward += reward

                self.step += 1

                # 5. update target network
                if self.step % self.target_update_step == 0:
                    if self.model == 'sac':
                        self.target_critic.load_state_dict(self.online_critic.state_dict())
                    else:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                self.logger.summary_scalar('train/reward', reward, self.step)
                self.logger.summary_hist('/train/logit', action_vec_, self.step)

                if i_episode % 2000 == 0:
                    self.logger.log('|{:04d}'.format(int(action_)), end='')
                    for idx in range(self.action_size):
                        self.logger.log('|{:04d}'.format(int(action_vec_[0, idx] * 10000)), end='')
                    self.logger.log('|')

                if loss is not None:
                    critic_loss = 0
                    actor_loss = 0
                    
                    if self.model == 'sac':
                        if isinstance(loss, list):
                            critic_loss, actor_loss = loss
                            self.logger.summary_scalar('train/critic_loss', critic_loss, self.step)
                            self.logger.summary_scalar('train/actor_loss', actor_loss, self.step)
                        self.logger.summary_scalar('train/dqn/opt_reward', reward_batch, self.step)
                        self.logger.summary_scalar('train/dqn/opt_q_value', q_batch, self.step)
                        self.logger.summary_scalar('train/dqn/opt_traget_q', traget_q_batch, self.step)

                    else:
                        self.logger.summary_scalar('train/dqn/opt_loss', loss, self.step)
                        self.logger.summary_scalar('train/dqn/opt_reward', reward_batch, self.step)
                        self.logger.summary_scalar('train/dqn/opt_q_value', q_batch, self.step)
                        self.logger.summary_scalar('train/dqn/opt_traget_q', traget_q_batch, self.step)
                else:
                    if self.model == 'sac':
                        critic_loss = 0
                        actor_loss = 0
                if done:
                    break

            # 6. validate the trained network
            if i_episode % self.play_episode_interval == 0:
                test_reward, test_score = self.play()
                self.save_checkpoint(filename=f'{self.save_folder}/checkpoints/chkpoint_{self.model}_{i_episode}.pth.tar')

                self.logger.summary_scalar('test/reward', test_reward, self.step)
                self.logger.summary_scalar('test/score', test_score, self.step)

                if self.best_reward <= int(np.mean(test_reward)):
                    self.logger.log(' %d th episode, eval play result : best reard : %.4f, test reward : %.4f' % (i_episode, self.best_reward , test_reward))
                    self.best_reward = int(np.mean(test_reward))
                    self.best_score = int(np.mean(test_score))
                    self.save_checkpoint(filename=f'{self.save_folder}/checkpoints/chkpoint_{self.model}_best_{self.best_reward}.pth.tar')

            if i_episode % 100 == 0 and i_episode != 0:
                loss = 0 if loss is None else loss
                if self.model == 'sac':
                    self.logger.log(' %d th episode, %d step, %.3f c-loss, %.3f a-loss, reward : %.4f' % (i_episode, self.step, critic_loss, actor_loss, ep_reward))
                else:
                    self.logger.log(' %d th episode, %d step, %.9f loss, reward : %.4f' % (i_episode, self.step , loss, ep_reward))
            self.cur_episode = i_episode

    ## Test trained agent according to test dataset
    # @brief Test trained agent according to test dataset
    # @param None
    # @return None
    def save_grid_image(self, input_img, pred_img, err_img, name_path, type='.jpg'):
        # img1 = (input_img * 255).astype(np.uint8).transpose(1, 2, 0)
        img1 = input_img.transpose(1, 2, 0)
        img2 = pred_img
        img3 = err_img

        if img2.shape[0] == 1 or len(img2.shape) == 2:
            img2 = np.repeat(np.expand_dims(img2, axis=0), 3, axis=0)
        img2 = (img2 * 255).astype(np.uint8).transpose(1, 2, 0)
        if img3.shape[0] == 1 or len(img3.shape) == 2:
            img3 = np.repeat(np.expand_dims(img3, axis=0), 3, axis=0)
        img3 = (img3 * 255).astype(np.uint8).transpose(1, 2, 0)

        if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
            img1 = imresize(img1, (img2.shape[0], img2.shape[1]))
        img1 = (img1 * 255).astype(np.uint8)
        save_img = np.concatenate([img1, img2, img3], axis=0)
        cv2.imwrite(name_path + type, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
        return save_img, img1, img2, img3

    def _to_numpy(self, img_tensor):
        img_np = (img_tensor * 255.0).cpu().numpy().astype(np.uint8).transpose(2, 1, 0)
        return img_np

    @torch.no_grad()
    def play(self): 
        # create folder if needed
        if self.is_save:
            save_path = self.save_folder + '/play/' + str(self.play_count).zfill(3) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if self.config.is_test:
                img_folder = self.save_folder + '/play/' + str(self.play_count).zfill(3) + '/image/'
                output_folder = self.save_folder + '/play/' + str(self.play_count).zfill(3) + '/output/'
                err_folder = self.save_folder + '/play/' + str(self.play_count).zfill(3) + '/error/'
                final_folder = self.save_folder + '/play/' + str(self.play_count).zfill(3) + '/final/'
                os.makedirs(img_folder, exist_ok=True)
                os.makedirs(output_folder, exist_ok=True)
                os.makedirs(err_folder, exist_ok=True)
                os.makedirs(final_folder, exist_ok=True)

        # loop for test batch
        test_done = False
        testset_idx = 0
        diction = {}
        pbar = tqdm.tqdm(desc='Playing ')
        save_play_count = [0, 1, 2, 3, 5]

        action_stat = [[0 for _ in range(self.action_size)] for _ in range(self.stop_step)]

        while not test_done:
            # initialize & get test images
            test_imgs, test_done, cur_scores, output_imgs, err_imgs = self.env.get_test_imgs(self.play_batch)  # current image No. and batch size
            if not isinstance(cur_scores, float):
                cur_scores = cur_scores.detach().cpu().numpy()
            for k, cur_score in zip(range(len(cur_scores)), cur_scores):
                idx = testset_idx + k
                if self.config.app == 'depth' and self.config.reward_func == 'all':
                    cur_score1 = cur_score[6]
                    diction[str(idx) + '_all_score'] = np.copy(np.expand_dims(cur_score, axis=0))
                else:
                    cur_score1 = cur_score
                diction[str(idx) +'_init_score'] = cur_score1

            if self.is_save:
                if self.play_count in save_play_count or self.play_count % 10 == 0 or self.config.is_test:
                    save_flag = True
                    for k, test_img, cur_score, output_img, err_img in zip(range(len(test_imgs)), test_imgs, cur_scores, output_imgs, err_imgs):
                        if self.config.app == 'depth' and self.config.reward_func == 'all':
                            cur_score1 = cur_score[6]
                        else:
                            cur_score1 = cur_score
                        C_, _, _ = test_img.shape
                        if C_ == 4:
                            test_img = bayer_to_rgb(test_img.unsqueeze(0)).squeeze(0)
                        idx = testset_idx + k
                        name = 'img_{:04d}_init_score_{}'.format(idx, cur_score1)
                        name_path = os.path.join(save_path, name)
                        save_img = test_img.detach().cpu().numpy()
                        save_img2 = output_img.detach().cpu().numpy()
                        save_img3 = err_img.detach().cpu().numpy()

                        if self.env.app == 'detection' or self.env.app == 'restoration':
                            pass
                        else:
                            save_img2 = np.clip(save_img2, 0, 1)
                            save_img3 = np.clip(save_img3, 0, 1)

                        all_img, img1, img2, img3 = self.save_grid_image(save_img, save_img2, save_img3, name_path)
                        if self.config.is_test:
                            img_path = os.path.join(img_folder, name + '.png')
                            cv2.imwrite(img_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                            output_path = os.path.join(output_folder, name + '.png')
                            cv2.imwrite(output_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                            err_path = os.path.join(err_folder, name + '.png')
                            cv2.imwrite(err_path, cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
                else:
                    save_flag = False
            else:
                save_flag = False
            # iteratively play
            finished = np.array([False for _ in range(self.play_batch)])

            for m in range(self.stop_step): 

                # get actions from network
                if self.model == 'sac':
                    with torch.no_grad():
                        all_actions = []
                        all_action_vec = []

                        actions, action_vec, _ = self.select_action(test_imgs, None, True)
                else :
                    actions_vec = self.policy_net(test_imgs)
                    actions = actions_vec.argmax(axis=1)

                # do action on image
                next_imgs, rewards, dones, next_scores, output_imgs, err_imgs = self.env.step_test(actions)

                # Save log and result
                actions = actions.detach().cpu().numpy()
                next_scores = next_scores.detach().cpu().numpy()

                # cur image, action (to do), reward (will get), score(cur image)
                for k, test_img, action, reward, done, cur_score, output_img, err_img, finish in zip(range(len(test_imgs)), test_imgs,
                                                                     actions, rewards, dones, cur_scores, output_imgs, err_imgs, finished) :
                    # store reward, score, action
                    idx = testset_idx + k
                    action_stat[m][int(action)] += 1
                    if self.config.app == 'depth' and self.config.reward_func == 'all':
                        cur_score1 = cur_score[6]
                        all_score = cur_score
                    else:
                        cur_score1 = cur_score
                        all_score = None

                    if finish:
                        reward = 0.0
                        action = 0.0
                        cur_score = diction[str(idx) + '_cur_score'][-1]

                    try:
                        if m == 0:
                            diction[str(idx) +'_reward']   = np.array([reward])
                            diction[str(idx) +'_action']   = np.array([action])
                            diction[str(idx) +'_cur_score'] = np.array([cur_score1])
                        else :
                            diction[str(idx) +'_reward']   = np.concatenate([diction[str(idx) +'_reward'], np.array([reward])], axis=0)
                            diction[str(idx) +'_action']   = np.concatenate([diction[str(idx) +'_action'], np.array([action])], axis=0)
                            diction[str(idx) +'_cur_score'] = np.concatenate([diction[str(idx) +'_cur_score'], np.array([cur_score1])], axis=0)
                        if all_score is not None:
                            diction[str(idx) + '_all_score'] = np.concatenate([diction[str(idx) + '_all_score'], np.copy(np.expand_dims(all_score, axis=0))], axis=0)
                    except:
                        print("error!!")
                    # save images
                    if self.is_save and save_flag:
                        # if random.random() < self.save_ratio:
                        name = 'img_{:04d}_step_{:02d}_reward_{:+.3f}_score_{:.3f}_{}_{}'.format(
                            idx, m,
                            float(reward),
                            float(cur_score1),
                            int(action),
                            self.env.tool_name[int(action)],

                        )
                        name_path = os.path.join(save_path, name)
                        C_, _, _ = test_img.shape
                        if C_ == 4:
                            test_img = bayer_to_rgb(test_img.unsqueeze(0)).squeeze(0)

                        save_img = test_img.clamp(0.0,1.0).detach().cpu().numpy()
                        save_img2 = output_img.clamp(0.0,1.0).detach().cpu().numpy()
                        save_img3 = err_img.clamp(0.0,1.0).detach().cpu().numpy()

                        if self.env.app == 'detection' or self.env.app == 'restoration':
                            pass
                        else:
                            save_img2 = np.clip(save_img2, 0, 1)
                            save_img3 = np.clip(save_img3, 0, 1)

                        all_img, img1, img2, img3 = self.save_grid_image(save_img, save_img2, save_img3, name_path)
                        if self.config.is_test:
                            img_path = os.path.join(img_folder, name + '.png')
                            cv2.imwrite(img_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                            output_path = os.path.join(output_folder, name + '.png')
                            cv2.imwrite(output_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                            err_path = os.path.join(err_folder, name + '.png')
                            cv2.imwrite(err_path, cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
                            if m == self.stop_step - 1:
                                if self.env.dataloader.last_test_name != '':
                                    name = self.env.dataloader.last_test_name.split('/')[-1]
                                else:
                                    name = f'{idx:04d}.png'
                                final_path = os.path.join(final_folder, name)
                                cv2.imwrite(final_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

                finished = np.logical_or(dones, finished)

                test_imgs = next_imgs
                cur_scores = next_scores

                # print results
                """
                print(('reward_' + str(m) + ': %.4f, score_' + str(m) + ': %.4f' +
                       ', tested images: %d, total tested images: %d') % (rewards.mean(), next_scores.mean(),
                                                                          len(test_imgs), testset_idx + len(test_imgs)))
                """
            testset_idx += len(test_imgs)
            pbar.update(test_imgs.shape[0])

        pbar.close()
        # print final results
        final_step_reward = 0
        final_step_score = 0

        self.logger.log('This is the final result:')
        total_reward = 0
        total_score = 0
        final_score = 0
        init_score = 0
        all_scores = None
        for idx in range(testset_idx):
            init_score += diction[str(idx) + '_init_score']
            if self.config.app == 'depth' and self.config.reward_func == 'all':
                if all_scores is None:
                    all_scores = diction[str(idx) + '_all_score']
                else:
                    all_scores += diction[str(idx) + '_all_score']
        if self.config.app == 'depth' and self.config.reward_func == 'all':
            all_scores /= testset_idx

        self.logger.log(f'Average init score: {init_score/testset_idx}')
        for m in range(self.stop_step):
            m_reward = 0 
            m_score = 0
            for idx in range(testset_idx):
                m_reward += diction[str(idx) + '_reward'][m]
                m_score  += diction[str(idx) + '_cur_score'][m]
            total_reward += m_reward
            total_score  += m_score
            self.logger.log((str(m) +'th Test reward' + ': %.4f, score' + str(self.stop_step) + ': %.4f') 
                % (m_reward/testset_idx, m_score/(testset_idx)))
            if m == self.stop_step - 1:
                final_score = m_score
        total_episodic_reward = 0
        for idx in range(testset_idx):
            episodic_reward = 0
            for m in range(self.stop_step):
                episodic_reward += diction[str(idx) + '_reward'][m]
            total_episodic_reward += episodic_reward
        self.logger.log(f'Average episodic reward: {total_episodic_reward / testset_idx}')
        final_step_reward = total_reward/testset_idx
        if self.stop_step == 0:
            final_step_score = 0
        else:
            final_step_score = final_score/testset_idx

        self.logger.log(('Test reward' + ': %.4f, score' + str(self.stop_step) + ': %.4f' +
               ', toal tested images: %d') % (final_step_reward,
                                              final_step_score, testset_idx))
        self.play_count += 1
        if self.config.app == 'depth' and self.config.reward_func == 'all':
            self.logger.log('all results:')
            table = ['absdif', 'absrel', 'sqrel', 'delta1', 'delta2', 'delta3', 'rmse', 'rmslog']
            self.logger.log(f'        ', end='')
            for metric in table:
                self.logger.log(f'  {metric:<6}  ', end='')
            self.logger.log('')
            self.logger.log(f'init  : ', end='')
            for idx in range(len(table)):
                self.logger.log('  {:6.4f}  '.format(all_scores[0, idx]), end='')
            self.logger.log('')
            for m in range(1, self.stop_step + 1):
                self.logger.log(f'step {m}: ', end='')
                for idx in range(len(table)):
                    self.logger.log('  {:6.4f}  '.format(all_scores[m, idx]), end='')
                self.logger.log('')

        if self.config.app == 'detection':
            results = self.env.dataloader.map_calc.get_results(save_path)

        if self.config.app == 'detection':
            self.logger.log('mAP results:')
            ious = results.keys()
            self.logger.log(f'        ', end='')
            for iou in ious:
                self.logger.log('  {:5.2f}  '.format(float(iou)), end='')
                # {results[0] * 100: .2f}
            self.logger.log('')
            self.logger.log(f'init  : ', end='')
            for iou in ious:
                self.logger.log('  {:5.2f}  '.format(results[iou][0] * 100), end='')
            self.logger.log('')
            for m in range(1, self.stop_step + 1):
                self.logger.log(f'step {m}: ', end='')
                for iou in ious:
                    self.logger.log('  {:5.2f}  '.format(results[iou][m]*100), end='')

                self.logger.log('')
        if self.config.is_test:
            self.logger.log('Action statistics')
            self.logger.log('         ', end='')
            for action_idx in range(self.action_size):
                self.logger.log(f'{action_idx:4d} ', end='')
            self.logger.log('')

            for step in range(self.stop_step):
                self.logger.log(f'step {step:02d}: ', end='')
                for action_idx in range(self.action_size):
                    self.logger.log(f'{action_stat[step][action_idx]:4d} ', end='')
                self.logger.log('')

        return final_step_reward, final_step_score  # m-th iteration result

    def save_checkpoint(self, filename='checkpoints/checkpoint.pth.tar'):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        if self.model in self.dqn_models :
            checkpoint = {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'step': self.step,
                'episode': self.cur_episode,
                'play_count': self.play_count,
                'best_reward': self.best_reward,
                'best_psnr': self.best_psnr,
                'best_score': self.best_score
            }

            checkpoint['optimizer'] = self.optimizer.state_dict()

        elif self.model == 'sac':
            checkpoint = {
                'policy_net': self.policy_net.state_dict(),
                'critic_net': self.online_critic.state_dict(),
                'target_critic_net': self.target_critic.state_dict(),
                'step': self.step,
                'episode': self.cur_episode,
                'play_count': self.play_count,
                'best_reward': self.best_reward,
                'best_psnr': self.best_psnr,
                'policy_optim': self.policy_optim.state_dict(),
                'best_score': self.best_score
            }
            if self.feature_extractor == 'cnn':
                checkpoint['q_optim'] = self.q_optim.state_dict()
            else:
                checkpoint['q1_optim'] = self.q1_optim.state_dict()
                checkpoint['q2_optim'] = self.q2_optim.state_dict()
        else:
            raise NotImplementedError
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename='checkpoints/checkpoint.pth.tar', epsilon=None):
        checkpoint = torch.load(filename)

        self.step = checkpoint['step']
        self.best_reward = self.best_reward or checkpoint['best_reward']
        self.best_score = checkpoint['best_score']
        self.best_psnr = checkpoint['best_psnr']
        if 'episode' in checkpoint:
            self.cur_episode = checkpoint['episode']
        if 'play_count' in checkpoint:
            self.play_count = checkpoint['play_count']
        if self.model in self.dqn_models:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])

            self.optimizer.load_state_dict(checkpoint['optimizer'])
        elif self.model == 'sac':
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.online_critic.load_state_dict(checkpoint['critic_net'])
            self.target_critic.load_state_dict(checkpoint['target_critic_net'])
            try:
                self.policy_optim.load_state_dict(checkpoint['policy_optim'])
            except:
                self.policy_optim.load_state_dict(checkpoint['policy_optim'].state_dict())
            try:
                if self.feature_extractor == 'cnn':
                    self.q_optim.load_state_dict(checkpoint['q_optim'])
                else:
                    self.q1_optim.load_state_dict(checkpoint['q1_optim'])
                    self.q2_optim.load_state_dict(checkpoint['q2_optim'])
            except:
                if self.feature_extractor == 'cnn':
                    self.q_optim.load_state_dict(checkpoint['q_optim'].state_dict())
                else:
                    self.q1_optim.load_state_dict(checkpoint['q1_optim'].state_dict())
                    self.q2_optim.load_state_dict(checkpoint['q2_optim'].state_dict())
