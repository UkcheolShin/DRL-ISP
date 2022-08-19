# Move individually trained tools into a single folder
mkdir -p ../tool_rl_selector/cnn_tool/syn_indiv/         
cp -prf outputs/syn_denoise/CNN_TOOL3/best.pth 	../tool_rl_selector/cnn_tool/syn_indiv/DeNoiseNet_L3.pth
cp -prf outputs/syn_denoise/CNN_TOOL8/best.pth 	../tool_rl_selector/cnn_tool/syn_indiv/DeNoiseNet_L8.pth
cp -prf outputs/syn_deblur/CNN_TOOL3/best.pth 	../tool_rl_selector/cnn_tool/syn_indiv/DeBlurNet_L3.pth
cp -prf outputs/syn_deblur/CNN_TOOL8/best.pth 	../tool_rl_selector/cnn_tool/syn_indiv/DeBlurNet_L8.pth
cp -prf outputs/syn_sr/CNN_TOOL3/best.pth		../tool_rl_selector/cnn_tool/syn_indiv/SRNet_L3.pth
cp -prf outputs/syn_sr/CNN_TOOL8/best.pth		../tool_rl_selector/cnn_tool/syn_indiv/SRNet_L8.pth
cp -prf outputs/syn_dejpeg/CNN_TOOL3/best.pth	../tool_rl_selector/cnn_tool/syn_indiv/DeJpegNet_L3.pth
cp -prf outputs/syn_dejpeg/CNN_TOOL8/best.pth	../tool_rl_selector/cnn_tool/syn_indiv/DeJpegNet_L8.pth
cp -prf outputs/syn_exposure/CNN_TOOL3/best.pth	../tool_rl_selector/cnn_tool/syn_indiv/ExposureNet_L3.pth
cp -prf outputs/syn_exposure/CNN_TOOL8/best.pth	../tool_rl_selector/cnn_tool/syn_indiv/ExposureNet_L8.pth
cp -prf outputs/syn_ctc/CNN_TOOL3/best.pth		../tool_rl_selector/cnn_tool/syn_indiv/CTCNet_L3.pth
cp -prf outputs/syn_ctc/CNN_TOOL8/best.pth		../tool_rl_selector/cnn_tool/syn_indiv/CTCNet_L8.pth
cp -prf outputs/syn_wb/CNN_TOOL3/best.pth		../tool_rl_selector/cnn_tool/syn_indiv/WBNet_L3.pth
cp -prf outputs/syn_wb/CNN_TOOL8/best.pth		../tool_rl_selector/cnn_tool/syn_indiv/WBNet_L8.pth

# joint training
CUDA_VISIBLE_DEVICES=2 python train_joint.py --dataset-dir /SSD1/Dataset_DRLISP_RAW/ --outputs-dir outputs/syn_joint/ --weight-dir ../tool_rl_selector/cnn_tool/syn_indiv/ --dataset syn --with-global --with-local --sequence_len 2
