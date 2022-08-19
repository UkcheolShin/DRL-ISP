# Move jointly trained tools into a single folder
mkdir -p ../tool_rl_selector/cnn_tool/syn_joint/          
cp -prf outputs/syn_joint/best_DeNoiseNet_L3.pth 	../tool_rl_selector/cnn_tool/syn_joint/DeNoiseNet_L3.pth
cp -prf outputs/syn_joint/best_DeNoiseNet_L8.pth 	../tool_rl_selector/cnn_tool/syn_joint/DeNoiseNet_L8.pth
cp -prf outputs/syn_joint/best_DeBlurNet_L3.pth 	../tool_rl_selector/cnn_tool/syn_joint/DeBlurNet_L3.pth
cp -prf outputs/syn_joint/best_DeBlurNet_L8.pth 	../tool_rl_selector/cnn_tool/syn_joint/DeBlurNet_L8.pth
cp -prf outputs/syn_joint/best_SRNet_L3.pth			../tool_rl_selector/cnn_tool/syn_joint/SRNet_L3.pth
cp -prf outputs/syn_joint/best_SRNet_L8.pth			../tool_rl_selector/cnn_tool/syn_joint/SRNet_L8.pth
cp -prf outputs/syn_joint/best_DeJpegNet_L3.pth		../tool_rl_selector/cnn_tool/syn_joint/DeJpegNet_L3.pth
cp -prf outputs/syn_joint/best_DeJpegNet_L8.pth		../tool_rl_selector/cnn_tool/syn_joint/DeJpegNet_L8.pth
cp -prf outputs/syn_joint/best_ExposureNet_L3.pth	../tool_rl_selector/cnn_tool/syn_joint/ExposureNet_L3.pth
cp -prf outputs/syn_joint/best_ExposureNet_L8.pth	../tool_rl_selector/cnn_tool/syn_joint/ExposureNet_L8.pth
cp -prf outputs/syn_joint/best_CTCNet_L3.pth		../tool_rl_selector/cnn_tool/syn_joint/CTCNet_L3.pth
cp -prf outputs/syn_joint/best_CTCNet_L8.pth		../tool_rl_selector/cnn_tool/syn_joint/CTCNet_L8.pth
cp -prf outputs/syn_wb/CNN_TOOL3/best.pth		    ../tool_rl_selector/cnn_tool/syn_joint/WBNet_L3.pth
cp -prf outputs/syn_wb/CNN_TOOL8/best.pth		    ../tool_rl_selector/cnn_tool/syn_joint/WBNet_L8.pth
