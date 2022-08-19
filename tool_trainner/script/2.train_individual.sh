# IROS Version
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_denoise/ --dataset syn --mode noise --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_denoise/ --dataset syn --mode noise --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_deblur/ --dataset syn --mode gusblur --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_deblur/ --dataset syn --mode gusblur --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_ctc/ --dataset syn --mode raw --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_ctc/ --dataset syn --mode raw --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_sr/ --dataset syn --mode sr --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_sr/ --dataset syn --mode sr --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_exposure/ --dataset syn --mode bright_gam --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_exposure/ --dataset syn --mode bright_gam --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_dejpeg/ --dataset syn --mode jpeg --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_dejpeg/ --dataset syn --mode jpeg --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_wb/ --dataset syn --mode wb --level low --network CNN_TOOL3 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_wb/ --dataset syn --mode wb --level high --network CNN_TOOL8 --num-channels 4 --net_skip --l1 --vgg_feat --augmentation 
