#!/bin/bash

source /home/wzy/miniconda3/etc/profile.d/conda.sh
cd ./dependencies/CIHP_PGN
conda activate tf

if ["" = $1]; then 
    img_dir=/home/wzy/workspace/avatar3D/sample_data/DCTON/test_img
else
    img_dir="../../"$1
fi

if ["" = $2]; then 
    out_dir=/home/wzy/workspace/avatar3D/sample_data/DCTON/test_pose
else
    out_dir="../../"$2
fi

python test_pgn.py --img-dir ${img_dir} --output-dir ${out_dir}
