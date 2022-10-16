import numpy as np
import pickle
import sys
import os
import torch.nn.functional as F
import imageio
import detectron2

### convert DensePose pickle into npy file ###
def convert_densepose(pkl_path, output_dir):
    sys.path.append('./dependencies/detectron2/projects/DensePose')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    for entry in data:
        file_path = entry['file_name']

        i = entry['pred_densepose'][0].labels.unsqueeze(0)
        uv = entry['pred_densepose'][0].uv

        ori_image = imageio.imread(file_path)
        h,w,_ = ori_image.shape

        i = F.interpolate(i.unsqueeze(0).float(), size=(h,w), mode='bilinear').squeeze(0).cpu().numpy()
        uv = F.interpolate(uv.unsqueeze(0).float(), size=(h,w), mode='bilinear').squeeze(0).cpu().numpy()

        iuv = np.concatenate((i,uv), axis=0)
        file_name = os.path.split(file_path)[-1].split('.')[0]
        np.save(f"{output_dir}/{file_name}.npy", iuv.astype(np.float32))

def get_keypoints(img_dir, output_dir):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import cv2
    import json

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl'

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
            print(arr)
            people_field = people_template.copy()
            people_field['pose_keypoints'] = list(arr.reshape(-1))
            json_string['people'].append(people_field)

        with open(f'{output_dir}/{img_name}_keypoints.json', 'w') as f:
            f.write(json.dumps(json_string))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-path', type=str, help="path to DensePose pickle output", default='./output/preprocess/results.pkl')
    parser.add_argument('--output-dir', type=str, help="dir to save converted files", default='./data/DCTON/')
    paser.add_argument('--img-dir', type=str, help="input image directory", default='/home/wzy/workspace/avatar3D/data/DCTON/test_img')
    args = parser.parse_args()

    convert_densepose(args.pkl_path, args.output_dir + 'test_densepose')

    get_keypoints(args.img_dir, args.output_dir + 'test_pose')