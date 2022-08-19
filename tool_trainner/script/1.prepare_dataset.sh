# 1. Synthetic dataset
echo "Prepare pre-synthetic dataset"
python data_loader/prepare_syn.py --images-dir-rgb /SSD1/Dataset_DRLISP/ --images-dir-raw /SSD1/Dataset_DRLISP_RAW/ --output-path /SSD1/Dataset_DRLISP_RAW/syn_dataset/  --patch-size 224 
python data_loader/prepare_syn.py --images-dir-rgb /SSD1/Dataset_DRLISP/ --images-dir-raw /SSD1/Dataset_DRLISP_RAW/ --output-path /SSD1/Dataset_DRLISP_RAW/syn_dataset/  --patch-size 224  --test 
