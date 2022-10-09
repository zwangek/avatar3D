import numpy as np
import pickle
import sys
import os
import argparse

### convert DensePose pickle into npy file ###
def convert_densepose(pkl_path, output_dir):
    sys.path.append('./dependencies/detectron2/projects/DensePose')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    for entry in data:
        filename = entry['file_name']
        filename = os.path.split(filename)[-1].split('.')[0]

        i = entry['pred_densepose'][0].labels.cpu().numpy()
        uv = entry['pred_densepose'][0].uv.cpu().numpy()

        iuv = np.concatenate((i.reshape(1, i.shape[0], i.shape[1]),uv), axis=0)
        np.save(f'{output_dir}/{filename}.npy', iuv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-path', type=str, help="path to DensePose pickle output", default='./output/preprocess/results.pkl')
    parser.add_argument('--output-dir', type=str, help="dir to save converted files", default='./data/DCTON/test_densepose')
    args = parser.parse_args()

    convert_densepose(args.pkl_path, args.output_dir)