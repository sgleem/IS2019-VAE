#!/bin/bash
# Copyright 2019    Yang Zhang
# Apache 2.0.

source path.sh

VOX_ROOT=${KALDI_ROOT}/egs/voxceleb/v2
train_dir=${VOX_ROOT}/exp/xvector_nnet_1a/xvectors_train
test_dir=${VOX_ROOT}/exp/xvector_nnet_1a/xvectors_voxceleb1_test
work_dir=`pwd`
# python3 -u main.py \
# 	--epoch 200 \
# 	--batch_size 200 \
# 	--n_hidden 1800 \
# 	--learn_rate 0.00001 \
# 	--beta1 0.5 \
# 	--dataset_path ./data/test/xvector.npz \
# 	--spk_path ./data/test/spk.npz \
# 	--z_dim 200 \
# 	--KL_weigth 0.03 \
# 	--cohesive_weight 10 \
# 	--is_training 0 

# wait

mkdir -p ${train_dir}
mkdir -p ${test_dir}

copy-vector ark:data/train/xvector_vae.ark ark,scp:${train_dir}/xvector.ark,${train_dir}/xvector.scp
copy-vector ark:data/test/xvector_vae.ark ark,scp:${test_dir}/xvector.ark,${test_dir}/xvector.scp

cd ${VOX_ROOT}
ivector-mean ark:data/train/spk2utt scp:${train_dir}/xvector.scp \
    ark,scp:${train_dir}/spk_xvector.ark,${train_dir}/spk_xvector.scp ark,t:${train_dir}/num_utts.ark || cd ${work_dir}
ivector-mean ark:data/voxceleb1_test/spk2utt scp:${test_dir}/xvector.scp \
    ark,scp:${test_dir}/spk_xvector.ark,${test_dir}/spk_xvector.scp ark,t:${test_dir}/num_utts.ark || cd ${work_dir}

cd ${work_dir}

echo "Copy complete"

# bash eer.sh
