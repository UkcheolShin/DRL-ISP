import cv2
import torch
from imageio import imread
import random
import copy
import os
import argparse
from path import Path
import torchvision.transforms as transforms

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.data.datasets.coco import load_coco_json
from detectron2.config import get_cfg
#from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

from reward.detection.calc_map import calc_map
from reward.detection.utils import *
from skimage.draw import rectangle, rectangle_perimeter
from reward.utils import raw4ch, bayer_to_rgb

import tqdm
## DRL-ISP Dataloader
# @brief RL Dataloader 
# @param config
# @return None
class Data_ObjDetection(object):
    def __init__(self, config):
        self.is_train = ~config.is_test
        self.train_root = Path(config.train_dir)
        self.test_root = Path(config.test_dir)
        self.original_train_len = 0
        self.used_train_len = 0
        self.original_test_len = 0
        self.used_test_len = 0
        self.dataset = 'coco'
        self.annType = 'bbox'
        # 1. Training dataset
        if self.is_train:
            # 1. Read training data
            label_path   = self.train_root/'annotations/instances_train2017.json'
            data_path    = self.train_root/'train2017/'
            dataset_name = 'coco_2017_train'

            # path/to/json, path/to/image_root /dataset_name
            data_label = load_coco_json(label_path, data_path, dataset_name)
            self.data_label = []
            no_gt_count = 0
            for k in range(len(data_label)):
                if data_label[k]['height'] == 480 and data_label[k]['width'] == 640:
                    train_data_gt = data_label[k]["annotations"]
                    if len(train_data_gt) > config.detection_gt_cut:
                        self.data_label.append(data_label[k])
                    else:
                        no_gt_count += 1

            # shuffle order
            random.seed(4)
            random.shuffle(self.data_label)
            self.original_train_len = len(self.data_label)
            self.used_train_len = len(self.data_label)

            if config.train_dataset_len != 0:
                train_len = config.train_dataset_len
                self.used_train_len = train_len
                self.data_label = self.data_label[:train_len]

            # 2. set data index
            self.train_data_index = 0
            self.train_data_len = len(self.data_label)

            # default HWC
            self.train_height, self.train_width, _ = imread(self.data_label[0]['file_name']).shape
        else:
            raise ValueError('Invalid dataset!')

        # 2. Test dataset
        # 1. Read training data
        label_path   = self.test_root/'annotations/instances_val2017.json'
        data_path    = self.test_root/'val2017/'
        dataset_name = 'coco_2017_val'

        # path/to/json, path/to/image_root /dataset_name
        test_data_label = load_coco_json(label_path, data_path, dataset_name)
        
        # remove unmatched height except 480.
        height_test = torch.zeros(len(test_data_label))
        self.test_data_label = []
        no_gt_count = 0
        for k in range(len(test_data_label)):
            if test_data_label[k]['height'] == 480 and test_data_label[k]['width'] == 640:
                test_data_gt = test_data_label[k]["annotations"]
                if len(test_data_gt) > config.detection_gt_cut:
                    self.test_data_label.append(test_data_label[k])
                else:
                    no_gt_count += 1

        self.original_test_len = len(self.test_data_label)
        self.used_test_len = len(self.test_data_label)

        # shuffle order
        random.seed(4)
        random.shuffle(self.test_data_label)
        if config.test_dataset_len >0:
            test_len = config.test_dataset_len
            self.used_test_len = test_len
            self.test_data_label  = self.test_data_label[:test_len]

        self.map_calc = mAP_Calc(self.test_data_label, config.episode_max_step)

        # current data index
        self.test_data_index = 0
        self.test_data_len = len(self.test_data_label)

        self.test_height, self.test_width, _ = imread(self.test_data_label[0]['file_name']).shape

        self.last_test_name = ''

    def ArrayToTensor(self, image):
        """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
        image = np.transpose(image, (2, 0, 1))
        tensor_img = torch.from_numpy(image).float()/255.
        return tensor_img

    def get_train_data(self) : 
        # if data index exceed the dataset range, load new dataset
        if self.train_data_index >= self.train_data_len:
            self.train_data_index = 0

        img = imread(self.data_label[self.train_data_index]["file_name"]).astype(np.float32)
        img = img.astype(np.float32)
        img = raw4ch(img)

        train_data = self.ArrayToTensor(img).unsqueeze(0)
        train_data_gt = self.data_label[self.train_data_index]["annotations"]

        self.train_data_index += 1
        return train_data, train_data_gt

    def get_test_data(self, batch_size=1):
        # read images from testset folder with batch number
        idx_start = self.test_data_index
        idx_end  = idx_start + batch_size
        if idx_start == 0:
            self.map_calc.reset_results()

        if(idx_end >= self.test_data_len) : 
            idx_end = self.test_data_len
            self.test_data_index = 0
            test_done = True
        else :
            self.test_data_index += batch_size
            test_done = False

        list_in_gt  = self.test_data_label[idx_start:idx_end]

        img_num   = len(list_in_gt)
        test_imgs = torch.FloatTensor(img_num, 4, self.test_height, self.test_width)
        test_gts  = []

        for k in range(img_num):        
            img = imread(list_in_gt[k]["file_name"]).astype(np.float32)
            img = img.astype(np.float32)
            img = raw4ch(img)

            test_imgs[k, ...] = self.ArrayToTensor(img).unsqueeze(0)
            test_gts.append(list_in_gt[k]["annotations"])

        return test_imgs, test_gts, test_done, [idx_start, idx_end]

    def get_data_shape(self):
        img = np.transpose(imread(self.test_data_label[0]['file_name']), (2, 0, 1))
        return img.shape


class mAP_Calc(object):
    def __init__(self, gt_data, episode_max_step):
        super(mAP_Calc, self).__init__()
        self.gt_data = gt_data
        self.gt_classes = []
        self.gt_counter_per_class = {}
        self.counter_images_per_class = {}
        self.bounding_boxes = {}
        self.episode_max_step = episode_max_step

        # self.analyze_gt()
        # self.reset_results()

    def analyze_gt(self):
        self.gt_classes = []
        self.gt_counter_per_class = {}
        self.counter_images_per_class = {}

        for one_img in self.gt_data:
            already_seen_classes = []
            for one_annotation in one_img['annotations']:
                class_key = self._to_class_key(one_annotation['category_id'])
                if class_key in self.gt_counter_per_class:
                    self.gt_counter_per_class[class_key] += 1
                else:
                    self.gt_counter_per_class[class_key] = 1

                if class_key not in already_seen_classes:
                    if class_key in self.counter_images_per_class:
                        self.counter_images_per_class[class_key] += 1
                    else:
                        self.counter_images_per_class[class_key] = 1
                    already_seen_classes.append(class_key)
        self.gt_classes = list(self.gt_counter_per_class.keys())
        self.gt_classes = sorted(self.gt_classes)

    def reset_results(self):
        self.bounding_boxes = [{} for _ in range(self.episode_max_step + 1)]

    def add_results(self, category_id, img_idx, confidence, bbox, step):
        class_key = self._to_class_key(category_id)
        img_key = str(img_idx)
        if img_key in self.bounding_boxes[step]:
            self.bounding_boxes[step][img_key].append(
                {
                    'category_id': class_key,
                    'confidence': confidence,
                    'bbox': bbox
                })
        else:
            self.bounding_boxes[step][img_key] = [
                {
                    'category_id': class_key,
                    'confidence': confidence,
                    'bbox': bbox
                }
            ]

    def get_results(self, result_base_dir):
        gt_dir = os.path.join(result_base_dir, 'gt')
        dr_dirs = [os.path.join(result_base_dir, f'dr{dr_idx}') for dr_idx in range(len(self.bounding_boxes))]
        os.makedirs(gt_dir, exist_ok=True)
        for one_dir in dr_dirs:
            os.makedirs(one_dir, exist_ok=True)

        for gt in self.gt_data:
            filename = '{:012d}.txt'.format(gt['image_id'])
            filepath = os.path.join(gt_dir, filename)
            with open(filepath, "w") as f:
                for annotation in gt['annotations']:
                    str1 = '{}'.format(self._to_class_key(annotation['category_id']))
                    bbox = copy.deepcopy(annotation['bbox'])
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    for bbox1 in bbox:
                        str1 += ' {}'.format(int(bbox1))
                    str1 += '\n'
                    f.write(str1)
            for step in range(len(self.bounding_boxes)):
                dr_filepath = os.path.join(dr_dirs[step], filename)
                with open(dr_filepath, "w") as f:
                    pass

        args = argparse.Namespace()
        args_dict = vars(args)
        args_dict['no_plot'] = True
        args_dict['quiet'] = True
        args_dict['ignore'] = None
        args_dict['set_class_iou'] = None
        args_dict['no_animation'] = True

        iou_list = np.arange(0.5, 1.0, 0.05)
        all_list = {}
        for iou in iou_list:
            results = []
            for step in range(len(self.bounding_boxes)):
                for img_idx in self.bounding_boxes[step].keys():
                    filename = '{:012d}.txt'.format(self.gt_data[int(img_idx)]['image_id'])
                    filepath = os.path.join(dr_dirs[step], filename)
                    with open(filepath, "w") as f:
                        for dr_idx, detection in enumerate(self.bounding_boxes[step][img_idx]):
                            str1 = '{}'.format(detection['category_id'])
                            str1 += ' {}'.format(detection['confidence'])
                            for bbox1 in detection['bbox']:
                                str1 += ' {}'.format(int(bbox1))
                            str1 += '\n'
                            f.write(str1)
                args_dict['gt_path'] = gt_dir
                args_dict['dr_path'] = dr_dirs[step]
                args_dict['output_path'] = os.path.join(result_base_dir, 'out')
                args.min_overlap = iou
                ap_dict, mAP = calc_map(args)
                results.append(mAP)

            all_list['{:.2f}'.format(iou)] = results

            with open(os.path.join(result_base_dir, 'result_{:.2f}.txt'.format(iou)), 'w') as f1:
                for mAP in results:
                    f1.write('{:.3f}\n'.format(mAP * 100))

        return all_list

    def _to_class_key(self, category_id):
        return '{:03d}'.format(category_id)


############################################################
## DRL-ISP Rewarder
# @brief RL Reward Generator 
# @param config
# @return None
class Rewarder_ObjDetection(object):
    def __init__(self, config, map_calc=None):
        self.device = config.device
        self.map_calc = map_calc
        # model load    
        cfg_file = './reward/detection/config/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml'

        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        cfg.INPUT.FORMAT = "RGB"         # note. all detectron model tak "BGR image", need to be convert
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

        cfg.freeze()

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.predictor = build_model(self.cfg)
        self.predictor.to(self.device)
        self.predictor.eval()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        checkpointer = DetectionCheckpointer(self.predictor)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        for param in self.predictor.parameters():
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
        self.metric_function = self._get_detection_metric
        self.use_small_detection = config.use_small_detection
        self.metric_setting = {
            'method': 'pr', # p: precission, r: recall
            'iou_threshold': 0.7,
            'small_image_multiplier': 2.0,
            'small_image_threshold': 4096,  # pixels, w * h
            'multiplier': [0.5, 0.5,]
        }
        self.reward_function = self._step_metric_reward
        self.get_playscore   = self.get_step_test_metric
        self.reward_scale    = config.reward_scaler

        self.cmap = self.labelcolormap(182)

        self.idx_range = [0, 0]

    ## DRL-ISP train param initialize
    # @brief 
    @torch.no_grad()
    def reset_train(self, train_data, train_data_gt) : 
        # initialize environmental param
        metric_init         = self.metric_function(train_data, train_data_gt)
        self.metric_init    = metric_init
        self.metric_prev    = metric_init
        return None

    ## DRL-ISP test param initialize
    # @brief 
    @torch.no_grad()
    def reset_test(self, test_data, test_data_gt, return_img=False, idx_range=None) :
        # initialize environmental param
        self.metric_init_test = torch.zeros(len(test_data))
        self.metric_prev_test = torch.zeros(len(test_data))
        imgs = []
        err_imgs = []
        self.idx_range = idx_range
        for k in range(len(test_data)):
            if return_img:
                init_metric, output_img, err_img, preds = self.metric_function(test_data[[k], ...], test_data_gt[k],
                                                                        return_img=True, return_preds=True)
                imgs.append(output_img)
                err_imgs.append(err_img)
            else:
                init_metric, preds = self.metric_function(test_data[[k], ...], test_data_gt[k], return_preds=True)

            img_idx = idx_range[0] + k

            for pred_class, pred_bbox, pred_confidence in zip(preds['instances']._fields['pred_classes'],
                                                              preds['instances']._fields['pred_boxes'],
                                                              preds['instances']._fields['scores']):
                category_id = int(pred_class.cpu().numpy())
                bbox = pred_bbox.cpu().numpy()
                confidence = float(pred_confidence.cpu().numpy())

                self.map_calc.add_results(category_id, img_idx, confidence, bbox, 0)

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
    def get_reward_test(self, data_in, data_gt, idx, return_img=False, img_step=0):
        if return_img:
            metric_cur, output_img, err_img, preds = self.metric_function(data_in, data_gt, return_img=True, return_preds=True)
        else:
            metric_cur, preds = self.metric_function(data_in, data_gt, return_img=False, return_preds=True)

        reward = self.reward_function(metric_cur, idx)
        done = self.done_function_test(metric_cur, idx)
        self.metric_prev_test[idx] = metric_cur

        img_idx = self.idx_range[0] + idx
        for pred_class, pred_bbox, pred_confidence in zip(preds['instances']._fields['pred_classes'],
                                                          preds['instances']._fields['pred_boxes'],
                                                          preds['instances']._fields['scores']):
            category_id = int(pred_class.cpu().numpy())
            bbox = pred_bbox.cpu().numpy()
            confidence = float(pred_confidence.cpu().numpy())

            self.map_calc.add_results(category_id, img_idx, confidence, bbox, img_step)

        if return_img:
            return reward, done, output_img, err_img
        return reward, done

    ## DRL-ISP Img restoration metric getter (test)
    # @brief 
    @torch.no_grad()
    def get_step_test_metric(self): 
        return self.metric_prev_test

    ## DRL-ISP Img restoration metric getter (test)
    # @brief 
    # https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
    def _draw_box(self, start, end, shape1):
        rr, cc = rectangle_perimeter((start[1], start[0]), (end[1], end[0]), shape=shape1, clip=True)
        rr1, cc1 = rectangle_perimeter((start[1] - 1, start[0] - 1), (end[1] - 1, end[0] - 1), shape=shape1, clip=True)
        rr2, cc2 = rectangle_perimeter((start[1] + 1, start[0] + 1), (end[1] + 1, end[0] + 1), shape=shape1, clip=True)
        # return np.concatenate([rr,  rr1], axis=0), np.concatenate([cc,  cc1], axis=0)
        return np.concatenate([rr, rr1, rr2], axis=0), np.concatenate([cc, cc1, cc2], axis=0)
        # return rr, cc

    def _calc_iou(self, bbox1, bbox2):  # x1y1x2y2
        # calc iou
        rect1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        rect2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        intersect_x = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
        intersect_y = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])

        if intersect_x <= 0 or intersect_y <= 0:
            return 0

        intersect_area = float(intersect_x * intersect_y)
        union_area = float(rect1_area + rect2_area - intersect_area)

        iou = intersect_area / union_area
        return iou

    def _get_precision(self, preds, gts, iou_threshold=0.5):
        pred_results = []
        results_small_image = []

        gt_used = [False for _ in range(len(gts))]
        if len(gts) == 0:
            return 0.0, 0.0

        for pred_class, pred_bbox in zip(preds['instances']._fields['pred_classes'],
                                         preds['instances']._fields['pred_boxes']):
            gt_results = []
            for gt_idx, one_gt in enumerate(gts):
                if gt_used[gt_idx]:
                    gt_results.append(0.0)
                    continue

                gt_class = int(one_gt[0])
                if gt_class != pred_class:
                    gt_results.append(0.0)
                    continue
                gt_bbox = copy.deepcopy(one_gt[1:5])
                gt_bbox[2:4] += gt_bbox[0:2]
                iou = self._calc_iou(pred_bbox, gt_bbox)
                gt_results.append(iou)

            max_iou = max(gt_results)
            max_index = gt_results.index(max_iou)

            if max_iou >= iou_threshold:
                pred_results.append(1)
                gt_used[max_index] = True
                gt_bbox = copy.deepcopy(gts[max_index][1:5])
                gt_bbox[2:4] += gt_bbox[0:2]

                gt_w = gt_bbox[2] - gt_bbox[0]
                gt_h = gt_bbox[3] - gt_bbox[1]
                if gt_w * gt_h <= self.metric_setting['small_image_threshold']:
                    results_small_image.append(1.0 * self.metric_setting['small_image_multiplier'])
                else:
                    results_small_image.append(1.0)

            else:
                pred_results.append(0)
                results_small_image.append(0.0)

        if len(pred_results) == 0:
            return 0.0, 0.0

        mean_score = sum(pred_results) / len(pred_results)
        small_score = sum(results_small_image) / len(results_small_image)
        return mean_score, small_score

    def _get_recall(self, preds, gts, iou_threshold=0.5):
        if len(preds['instances']) == 0:
            return 0.0, 0.0

        results = []
        results_small_image = []
        pred_used = [False for _ in range(len(preds['instances']))]

        for gt_idx, one_gt in enumerate(gts):
            pred_results = []
            gt_class = int(one_gt[0])
            gt_bbox = copy.deepcopy(one_gt[1:5])
            gt_bbox[2:4] += gt_bbox[0:2]

            for pred_idx, pred_class, pred_bbox in zip(range(len(preds['instances'])),
                                                       preds['instances']._fields['pred_classes'],
                                                       preds['instances']._fields['pred_boxes']):
                if pred_used[pred_idx]:
                    pred_results.append(0.0)
                    continue
                if gt_class != pred_class:
                    pred_results.append(0.0)
                    continue

                iou = self._calc_iou(pred_bbox, gt_bbox)
                pred_results.append(iou)
            max_iou = max(pred_results)
            max_index = pred_results.index(max_iou)
            if max_iou >= iou_threshold:
                results.append(1)
                pred_used[max_index] = True
                gt_w = gt_bbox[2] - gt_bbox[0]
                gt_h = gt_bbox[3] - gt_bbox[1]
                if gt_w * gt_h <= self.metric_setting['small_image_threshold']:
                    results_small_image.append(1.0 * self.metric_setting['small_image_multiplier'])
                else:
                    results_small_image.append(1.0)
            else:
                results.append(0)
                results_small_image.append(0.0)

        if len(results) == 0:
            return 0.0, 0.0

        mean_score = sum(results) / len(results)
        small_score = sum(results_small_image) / len(results_small_image)
        return mean_score, small_score

    @torch.no_grad()
    def _get_detection_precision(self, img_in, label_gt, return_img=False, iou_threshold=0.5):
        height, width = img_in.shape[2:4]  # BCHW
        if return_img:
            output_img = np.copy(img_in.detach().cpu().numpy())
            err_img = np.copy(img_in.detach().cpu().numpy())

        img_in_np = (img_in * 255)[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        img_in_ = self.aug.get_transform(img_in_np).apply_image(img_in_np)
        img_in = torch.as_tensor(img_in_.astype('float32').transpose(2, 0, 1))
        inputs = {"image": img_in, "height": height, "width": width}
        with torch.no_grad() :
            preds = self.predictor([inputs])[0]

        label_gt_ = torch.zeros(len(label_gt),5)
        for k, label in zip(range(len(label_gt)), label_gt) :
            label_gt_[k, 0] = label_gt[k]['category_id'] # label
            label_gt_[k, 1:] = torch.Tensor(label_gt[k]['bbox']) # xywh
            if return_img:
                gt_bbox = label_gt_[k, 1:5]
                gt_bbox = gt_bbox.numpy().astype(np.uint16).tolist()
                color = self.cmap[label_gt[k]['category_id']]
                err_img = self._draw_bbox(err_img, gt_bbox, color)

        if len(preds['instances']) == 0:
            mean_score = 0
        else:
            mean_score = self._get_precision(preds, label_gt_, iou_threshold)

        if return_img:
            return mean_score, torch.from_numpy(output_img), torch.from_numpy(err_img)
        return mean_score

    @torch.no_grad()
    def _get_detection_metric(self, img_in, label_gt, return_img=False, return_preds=False):
        method = self.metric_setting['method']
        iou_threshold = self.metric_setting['iou_threshold']
        multiplier = self.metric_setting['multiplier']
        _, C_, height, width = img_in.shape
        if C_ == 4:
            img_in = bayer_to_rgb(img_in)

        if return_img:
            output_img = np.copy(img_in.detach().cpu().numpy())
            err_img = np.copy(img_in.detach().cpu().numpy())

        img_in_np = (img_in * 255)[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        img_in_ = self.aug.get_transform(img_in_np).apply_image(img_in_np)
        img_in = torch.as_tensor(img_in_.astype('float32').transpose(2, 0, 1))
        inputs = {"image": img_in, "height": height, "width": width}
        with torch.no_grad() :
            preds = self.predictor([inputs])[0]

        label_gt_ = torch.zeros(len(label_gt),5)
        for k, label in zip(range(len(label_gt)), label_gt) :
            label_gt_[k, 0] = label_gt[k]['category_id'] # label
            label_gt_[k, 1:] = torch.Tensor(label_gt[k]['bbox']) # xywh
            if return_img:
                gt_bbox = copy.deepcopy(label_gt_[k, 1:5])
                gt_bbox[2:4] += gt_bbox[0:2]
                gt_bbox = gt_bbox.numpy().astype(np.uint16).tolist()
                gt_w = gt_bbox[2] - gt_bbox[0]
                gt_h = gt_bbox[3] - gt_bbox[1]
                color = self.cmap[label_gt[k]['category_id']]
                err_img = self._draw_bbox(err_img, gt_bbox, color)
        label_gt_ = label_gt_.to(self.device)
        results = []
        mean_score = 0
        if return_img:
            if len(preds['instances']) == 0:
                mean_score = 0
            else:
                for k in range(len(preds['instances'])):
                    pred_bbox = preds['instances'][k].pred_boxes.tensor.squeeze(0).cpu().numpy()
                    color = self.cmap[preds['instances']._fields['pred_classes'][k]]
                    output_img = self._draw_bbox(output_img, pred_bbox, color)

        if 'p' in method:
            idx = method.index('p')
            multi = multiplier[idx]
            precision, small_precision = self._get_precision(preds, label_gt_, iou_threshold)
            precision *= multi
            small_precision *= multi
            if self.use_small_detection:
                score = small_precision
            else:
                score = precision
            mean_score += score

        if 'r' in method:
            idx = method.index('r')
            multi = multiplier[idx]
            recall, small_recall = self._get_recall(preds, label_gt_, iou_threshold)
            recall *= multi
            small_recall *= multi
            if self.use_small_detection:
                score = small_recall
            else:
                score = recall
            mean_score += score

        return_arr = [mean_score]

        if return_img:
            return_arr += [torch.from_numpy(output_img), torch.from_numpy(err_img)]
        if return_preds:
            return_arr += [preds]
        if len(return_arr) == 1:
            return_arr = return_arr[0]
        return return_arr

    def _draw_bbox(self, img, bbox, color):
        rr, cc = self._draw_box(bbox[0:2], bbox[2:4], img.shape[2:])
        img_r = img[0, 0]
        img_g = img[0, 1]
        img_b = img[0, 2]
        img_r[rr, cc] = color[0]
        img_g[rr, cc] = color[1]
        img_b[rr, cc] = color[2]
        return img

    @torch.no_grad()
    def _get_detection_iou(self, img_in, label_gt, return_img=False):
        height, width = img_in.shape[2:4]  # BCHW
        if return_img:
            output_img = np.copy(img_in.detach().cpu().numpy())
            err_img = np.copy(img_in.detach().cpu().numpy())

        img_in_np = (img_in * 255)[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        img_in_ = self.aug.get_transform(img_in_np).apply_image(img_in_np)
        img_in = torch.as_tensor(img_in_.astype('float32').transpose(2, 0, 1))
        inputs = {"image": img_in, "height": height, "width": width}
        with torch.no_grad() :
            preds = self.predictor([inputs])[0]

        label_gt_ = torch.zeros(len(label_gt),5)
        for k, label in zip(range(len(label_gt)), label_gt) :
            label_gt_[k, 0] = label_gt[k]['category_id'] # label
            label_gt_[k, 1:] = torch.Tensor(label_gt[k]['bbox']) # xywh
            if return_img:
                gt_bbox = copy.deepcopy(label_gt_[k, 1:5])
                gt_bbox[2:4] += gt_bbox[0:2]
                gt_bbox = gt_bbox.numpy().astype(np.uint16).tolist()
                err_img = self._draw_bbox(err_img, gt_bbox, (0, 1, 0))

        if len(preds['instances']) == 0:
            mean_score = 0
        else:
            mean_score = 0
            pred_results = []

            for pred_class, pred_bbox in zip(preds['instances']._fields['pred_classes'],
                                        preds['instances']._fields['pred_boxes']):
                gt_results = []
                for one_gt in label_gt_:
                    gt_class = int(one_gt[0])
                    gt_bbox = copy.deepcopy(one_gt[1:5])
                    pred_class = int(pred_class)

                    if gt_class != pred_class:
                        gt_results.append(0)
                        continue
                    gt_bbox[2:4] += gt_bbox[0:2]
                    iou = self._calc_iou(pred_bbox, gt_bbox)
                    gt_results.append(iou)

                pred_results.append(max(gt_results))

                if return_img:
                    output_img = self._draw_bbox(output_img, pred_bbox.cpu().numpy().astype(np.uint16).tolist(),
                                                 (1, 0, 0))

            mean_score = np.mean(np.array(pred_results))

        if return_img:
            return mean_score, torch.from_numpy(output_img), torch.from_numpy(err_img)
        return mean_score

    @torch.no_grad()
    def _get_detection_error(self, img_in, label_gt, return_img=False):

        height, width = img_in.shape[2:4] #BCHW
        if return_img:
            output_img = np.copy(img_in.detach().cpu().numpy())
            err_img = np.copy(img_in.detach().cpu().numpy())

        img_in_np = (img_in * 255)[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)
        img_in_ = self.aug.get_transform(img_in_np).apply_image(img_in_np)
        img_in = torch.as_tensor(img_in_.astype('float32').transpose(2, 0, 1))
        inputs = {"image": img_in, "height": height, "width": width}
        with torch.no_grad() :
            preds = self.predictor([inputs])[0]


        label_gt_ = torch.zeros(len(label_gt),5)
        for k, label in zip(range(len(label_gt)), label_gt) : 
            label_gt_[k, 0] = label_gt[k]['category_id'] # label
            label_gt_[k, 1:] = torch.Tensor(label_gt[k]['bbox']) # xywh
            if return_img:
                rr, cc = self._draw_box(label_gt_[k, 1:3].numpy().astype(np.uint16).tolist(),
                                        (label_gt_[k, 1:3] + label_gt_[k, 3:5]).numpy().astype(np.uint16).tolist(),
                                        err_img.shape[2:])
                img_b = err_img[0, 2]
                img_g = err_img[0, 1]
                img_r = err_img[0, 0]
                img_b[rr, cc] = 1.0
                img_g[rr, cc] = 0.0
                img_r[rr, cc] = 0.0

        if len(preds['instances']) == 0:
            mean_score = 0
        else:
            label_pred_ = torch.zeros(len(preds['instances']), 6)
            for k in range(len(label_pred_)) :
                label_pred_[k, :4] =  preds['instances'][k].pred_boxes.tensor # x1y1x2y2
                label_pred_[k, 4]  =  preds['instances'][k].scores       # score
                label_pred_[k, 5]  =  preds['instances'][k].pred_classes # label
                if return_img:
                    rr, cc = self._draw_box(label_pred_[k, 0:2].numpy().astype(np.uint16).tolist(),
                                            label_pred_[k, 2:4].numpy().astype(np.uint16).tolist(),
                                            output_img.shape[2:])
                    img_r = output_img[0, 0]
                    img_g = output_img[0, 1]
                    img_b = output_img[0, 2]
                    img_r[rr, cc] = 1.0
                    img_g[rr, cc] = 0.0
                    img_b[rr, cc] = 0.0

                    # img_r = err_img[0, 0]
                    # img_r[rr, cc] = 1.0
            mean_score = label_pred_[:,4].sum()/(len(label_pred_[:,4]) +1e-7)
        #sample_metrics = get_batch_statistics(label_pred_, label_gt_, iou_threshold=0.5)

        if return_img:
            return mean_score, torch.from_numpy(output_img), torch.from_numpy(err_img)
        return mean_score

    def uint82bin(self, n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    def labelcolormap(self, N):
        if N == 35:  # cityscape
            cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                             (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                             (102, 102, 156), (190, 153, 153),
                             (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                             (250, 170, 30), (220, 220, 0),
                             (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                             (0, 0, 70),
                             (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                             (0, 0, 142)],
                            dtype=np.uint8)
        else:
            cmap = np.zeros((N, 3), dtype=np.uint8)
            for i in range(N):
                r, g, b = 0, 0, 0
                id = i + 1  # let's give 0 a color
                for j in range(7):
                    str_id = self.uint82bin(id)
                    r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                    g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                    b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                    id = id >> 3
                cmap[i, 0] = r
                cmap[i, 1] = g
                cmap[i, 2] = b
        cmap = cmap.astype(np.float) / 255.0
        return cmap

    ## DRL-ISP inplace reward function
    # @brief decide done flag according to metric difference, episode_step
    @torch.no_grad()
    def _step_metric_reward(self, metric_cur, idx=None):
        if idx == None : 
            reward = metric_cur - self.metric_prev  # if RMSE error, cur error is lower than prev, i.e, prev_err > cur_err
        else:
            reward = metric_cur - self.metric_prev_test[idx] # if RMSE error, cur error is lower than prev, i.e, prev_err > cur_err
        return self.reward_scale * reward


    ## DRL-ISP inplace done function
    # @brief decide done flag according to metric difference, episode_step
    @torch.no_grad()
    def _done_function(self, metric_cur):
        if self.metric_init - metric_cur < -100 :  # RMSE is too bad, stop.
            return True
        return False

    ## DRL-ISP inplace done function
    # @brief decide done flag according to metric difference, episode_step
    @torch.no_grad()
    def _done_function_with_idx(self, metric_cur, idx):
        if self.metric_init_test[idx] - metric_cur< -100 :  # metric is too bad
            return True
        return False

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
    args.train_dir = '/DB/Dataset_DRLISP/MS_COCO/'

    # check data loader
    dataloader     = Data_ObjDetection(args)
    train_img, train_gt = dataloader.get_train_data()
    print('train img shape :' + str(train_img.shape)); print('train gt shape :' + str(len(train_gt)))
    test_imgs, test_gts, done = dataloader.get_test_data(32)
    print('test img shape :' + str(test_imgs.shape)); print('test gt shape :' + str(len(test_gts)));  print('test done :' + str(done))
    print('dataloader.get_dat_shape :' + str(dataloader.get_data_shape()))
    
    rewarder       = Rewarder_ObjDetection(args)
    rewarder.reset_train(train_img.to(device), train_gt)
    rewarder.reset_test(test_imgs.to(device), test_gts)

    train_reward, done = rewarder.get_reward_train(train_img.to(device), train_gt)
    print('train_reward:' + str(train_reward)); print('train done:' + str(done)); 
    idx=1
    test_rewards, done = rewarder.get_reward_test(test_imgs[[idx]].to(device), test_gts[idx], idx)
    print('test_reward:' + str(test_rewards)); print('test done:' + str(done)); 
    cur_score = rewarder.get_step_test_metric()
    print('test batch cur reward :' + str(cur_score)); print('test idx reward:' + str(cur_score[idx])); 
    