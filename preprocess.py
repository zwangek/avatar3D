import numpy as np
import pickle
import sys
import os
import argparse
import torch.nn.functional as F
import imageio

### convert DensePose pickle into npy file ###
def convert_densepose(pkl_path, output_dir):
    sys.path.append('./dependencies/DensePose')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    for entry in data:
        filename = entry['file_name']

        i = entry['pred_densepose'][0].labels.unsqueeze(0)
        uv = entry['pred_densepose'][0].uv

        ori_image = imageio.imread(filename)
        h,w,_ = ori_image.shape

        i = F.interpolate(i.unsqueeze(0).float(), size=(h,w), mode='bilinear').squeeze(0).cpu().numpy()
        uv = F.interpolate(uv.unsqueeze(0).float(), size=(h,w), mode='bilinear').squeeze(0).cpu().numpy()


        iuv = np.concatenate((i,uv), axis=0)
        np.save(f"{output_dir}/{os.path.split(filename)[-1].split('.')[0]}.npy", iuv.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-path', type=str, help="path to DensePose pickle output", default='./output/preprocess/results.pkl')
    parser.add_argument('--output-dir', type=str, help="dir to save converted files", default='./data/DCTON/test_densepose')
    args = parser.parse_args()

    convert_densepose(args.pkl_path, args.output_dir)