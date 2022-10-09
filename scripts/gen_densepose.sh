input=/home/wzy/workspace/avatar3D/data/DCTON/test_img/000129_0.jpg
pkl_path=./output/preprocess/results.pkl

python ./dependencies//DensePose/apply_net.py dump \
./dependencies//DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC1M_s1x.yaml \
https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_WC1M_s1x/216245771/model_final_0ebeb3.pkl \
${input} --output ${pkl_path} -v

python preprocess.py --pkl-path ${pkl_path} --output-dir ./data/DCTON/test_densepose/


