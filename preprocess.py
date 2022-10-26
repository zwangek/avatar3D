import json
import os

import cv2
import numpy as np
import torch.nn.functional as F
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def get_denspose(img_dir, output_dir):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file('/home/wzy/workspace/avatar3D/dependencies/detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC1M_s1x.yaml')
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC1M_s1x/216245771/model_final_0ebeb3.pkl'
    
    predictor = DefaultPredictor(cfg)
    extractor = DensePoseResultExtractor()

    for img_file in os.listdir(img_dir):
        image_name, postfix = img_file.split('.')
        img = cv2.imread(img_dir + '/' + img_file)
        ih,iw,ic = img.shape

        outputs = predictor(img)['instances']
        dp, bbox_xywh = (item[0] for item in extractor(outputs))
        x,y,w,h = (int(i) for i in bbox_xywh)

        i = dp.labels.unsqueeze(0).cpu().numpy()

        ### gen color map as vitonhd
        # label_map = np.zeros((ih, iw))
        # mask_bg = np.zeros((ih,iw))
        # mask_bg[y:y+h, x:x+w][i[0]!=0] = 1 # non zero region
        # mask_bg = np.tile((mask_bg==0)[:, :, np.newaxis], [1, 1, 3])
        # label_map[y:y+h, x:x+w] = i[0] * 255 / 24
        # label_map = cv2.applyColorMap(label_map.astype(np.uint8), cv2.COLORMAP_PARULA)
        # label_map[mask_bg] = 0
        # cv2.imwrite('./test.png', label_map)

        ### gen label as dcton
        # label = np.zeros((ih,iw))
        # label[y:y+h, x:x+w] = i[0]
        # cv2.imwrite('./test_label.png', label.astype(np.uint8))

        uv = dp.uv.cpu().numpy()
        iuv = np.concatenate((i,uv), axis=0)

        iuv = np.pad(iuv, ((0,0), (y, ih-h-y), (x, iw-w-x)), mode='constant', constant_values=0)
        np.save(f"{output_dir}/{image_name}.npy", iuv.astype(np.float32))

def get_keypoints(img_dir, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml')

    predictor = DefaultPredictor(cfg)

    for img_file in os.listdir(img_dir):
        img_name, postfix = img_file.split('.')
        if postfix not in ['png', 'jpg']:
            continue
        img = cv2.imread(img_dir + '/' + img_file)
        outputs = predictor(img)
        keypoints = outputs['instances'].pred_keypoints.cpu().numpy()

        json_string = {
            "version": 1.0,
            "people": [],
        }

        people_template = {
            "face_keypoints": [],
            "pose_keypoints": [],
            "hand_right_keypoints": [],
            "hand_left_keypoints": []
        }

        def keypoint_coco_to_18(coco):
            coco_dict = {
                'nose': 0,
                'left_eye': 1,
                'right_eye': 2,
                'left_ear': 3,
                'right_ear': 4,
                'left_shoulder': 5,
                'right_shoulder': 6,
                'left_elbow': 7,
                'right_elbow': 8,
                'left_wrist': 9,
                'right_wrist': 10,
                'left_hip': 11,
                'right_hip': 12,
                'left_knee': 13,
                'right_knee': 14,
                'left_ankle': 15,
                'right_ankle': 16,
            }
            kp18_dict = {
                'nose': 0,
                'right_eye': 14,
                'left_eye': 15,
                'right_ear': 16,
                'left_ear': 17,
                'right_shoulder': 2,
                'left_shoulder': 5,
                'right_elbow': 3,
                'left_elbow': 6,
                'right_wrist': 4,
                'left_wrist': 7,
                'right_hip': 8,
                'left_hip': 11,
                'right_knee': 9,
                'left_knee': 12,
                'right_ankle': 10,
                'left_ankle': 13,
            }
            arr = np.zeros((18,3))

            for k in coco_dict.keys():
                arr[kp18_dict[k]] = coco[coco_dict[k]]
        
            arr[1] = (arr[kp18_dict['left_shoulder']] + arr[kp18_dict['right_shoulder']]) / 2
            return arr

        for people in keypoints:
            arr = keypoint_coco_to_18(people)
            people_field = people_template.copy()
            people_field['pose_keypoints'] = list(arr.reshape(-1))
            json_string['people'].append(people_field)

        with open(f'{output_dir}/{img_name}_keypoints.json', 'w') as f:
            f.write(json.dumps(json_string))

def center_crop(img, target_size=(256,192)):
    h,w = target_size
    ih,iw,ic = img.shape
    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, help="dir to save converted files", default='./sample_data/DCTON/')
    parser.add_argument('--img-dir', type=str, help="input image directory", default='./sample_data/DCTON/test_img')
    args = parser.parse_args()

    print('generating densepose')
    get_denspose(args.img_dir, args.output_dir + 'test_densepose')

    print('generating keypoints')
    get_keypoints(args.img_dir, args.output_dir + 'test_pose')

    print('generating parse')
    os.system(f'bash ./scripts/gen_parse.sh {args.img_dir} {args.output_dir}/test_label' )

