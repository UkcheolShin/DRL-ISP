import argparse
import time
import random
import numpy as np
import json

# torch library
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# logger library
from logger import TermLogger, AverageMeter

from agent import Agent
from environment import MyEnvironment

# argument parsing
parser = argparse.ArgumentParser(description='pytorch implementation for RL-restore',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# network parameter 
parser.add_argument('--use_cpu', action='store_true', help='Whether to use gpu or not')
parser.add_argument('--is_test', action='store_true', help='Whether to do training or testing')
parser.add_argument('--agent', type=str, choices=['dqn_cnn', 'dqn_hist', 'sac'], default='sac')
parser.add_argument('--app', type=str, choices=['detection', 'depth', 'restoration'], default='restoration')
parser.add_argument('--reward_func', type=str, default='all', help='reward function')
parser.add_argument('--train_episodes', default=400000, type=int, help='Total training episode')
parser.add_argument('--memory_size', default=100000, type=int, metavar='N', help='n')
parser.add_argument('--learn_start', default=25000, type=int, metavar='N', help='n')
parser.add_argument('--test_step', default=5000, type=int, metavar='N', help='n')
parser.add_argument('--save_step', default=50000, type=int, metavar='N', help='n')
parser.add_argument('--max_step', default=2000000, type=int, metavar='N', help='n')
parser.add_argument('--target_q_update_step', default=5000, type=int, metavar='N', help='n')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='n')
parser.add_argument('--episode_max_step', default=3, type=int, metavar='N', help='n')
parser.add_argument('--cur_episode', default=-1, type=int)
parser.add_argument('--play_count', default=-1, type=int)
parser.add_argument('--continue_file', default='', help='Continue training')
parser.add_argument('--is_save', action='store_true', help='Whether to save results')
parser.add_argument('--log_dir', type=str, default='logs/', help='Path for logs')
parser.add_argument('--save_dir', type=str, default='./results/', help='Path for saving models and playing result')
parser.add_argument('--train_dir', type=str, default='dataset/train/', help='Path for training model')
parser.add_argument('--test_dir', type=str, default='dataset/test/', help='Path for testing model')
parser.add_argument('--resize_denominator', default=1.0, type=float, help='image resize denominator')
parser.add_argument('--clear_previous_result', action='store_true', help='Remove previous results')
parser.add_argument('--use_tool_factor', action='store_true', help='Use various tool factors')
parser.add_argument('--not_use_trad_tools', action='store_true')
parser.add_argument('--not_use_cnn_tools', action='store_true')
parser.add_argument('--layers', default=2, type=int, help='Layer number')
parser.add_argument('--layer_hidden', default=256, type=int, help='Layer hidden')
parser.add_argument('--layer_activation', default='relu', choices=['relu', 'tanh', 'sigmoid'], help='Layer activation')
parser.add_argument('--reward_scaler', default=0.1, type=float)
parser.add_argument('--reward_clip', default=2.0, type=float)
parser.add_argument('--train_dataset_len', default=0, type=int)
parser.add_argument('--test_dataset_len', default=0, type=int)
parser.add_argument('--toolpath', default='cnn_tool/syn_joint/')
parser.add_argument('--detection_gt_cut', default=5, type=int)
parser.add_argument('--use_small_detection', action='store_true')
parser.add_argument('--per', action='store_true')

# testing parameter
parser.add_argument('--dataset', type=str, choices=['ETH', 'syn'], default='syn', help='the dataset to train')
parser.add_argument('--play_model', type=str, default='./models/', help='Path for testing model')
parser.add_argument('--play_batch', default=1, type=int, metavar='N', help='n')
parser.add_argument('--metric', default='psnr', help='feature extractor', choices=['psnr', 'ssim', 'msssim'])

#SAC
parser.add_argument('--noise_clip', default=0.5, type=float, help='noise clip')
parser.add_argument('--policy_noise', default=0.1, type=float, help='policy noise')
parser.add_argument('--start_timestep', default=25000, type=int, help='random action time step')
parser.add_argument('--target_entropy_ratio', default=0.98, type=float, help='target entropy')
parser.add_argument('--policy_freq', default=2, type=int, help='delayed policy updates')
parser.add_argument('--sac_alpha', default=0.02, type=float, help='sac alpha')
parser.add_argument('--feature_extractor', default='hist', help='feature extractor', choices=['hist', 'cnn', 'rand'])

parser.add_argument('--not_use_inten', action='store_true')
parser.add_argument('--not_use_grad', action='store_true')
parser.add_argument('--not_use_seman', action='store_true')
parser.add_argument('--seman_net_name', default='AlexNet', choices=['AlexNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

def main():
    global device
    args = parser.parse_args()
    defaults = {}
    for key in vars(args):
        defaults[key] = parser.get_default(key)

    args.device = device

    env = MyEnvironment(args)
    agent = Agent(args, env, defaults)

    if not args.is_test:
        agent.train()
    else:
        agent.play()


if __name__ == '__main__':
    main()
