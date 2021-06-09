#!/bin/bash

# Author: Yang Zhang
# Author: Xueyi Wang
# Apache 2.0.
# 2019, CSLT

source path.sh

VOX_ROOT=${KALDI_ROOT}/egs/voxceleb/v2
train_dir=${VOX_ROOT}/exp/xvector_nnet_1a/xvectors_train
test_dir=${VOX_ROOT}/exp/xvector_nnet_1a/xvectors_voxceleb1_test

work_dir=`pwd`
# xvector
if [ ! -e data/train/xvector.npz ]; then
	mkdir -p data/train
	cd ${train_dir}/../../..
	copy-vector scp:${train_dir}/xvector.scp ark,t:${work_dir}/data/train/xvector.ark || cd ${work_dir}
	cd ${work_dir}
	python3 -u data/zip.py --source_path data/train/xvector.ark --dest_path data/train/xvector.npz
	echo
fi 
if [ ! -e data/test/xvector.npz ]; then
	mkdir -p data/test
	cd ${test_dir}/../../..
	copy-vector scp:${test_dir}/xvector.scp ark,t:${work_dir}/data/test/xvector.ark || cd ${work_dir}
	cd ${work_dir}
	python3 -u data/zip.py --source_path data/test/xvector.ark --dest_path data/test/xvector.npz
	echo
fi 

if [ -e data/train/xvector.ark ]; then
	rm data/train/xvector.ark
fi
if [ -e data/test/xvector.ark ]; then
	rm data/test/xvector.ark
fi

# utt2spk
train_dir=${VOX_ROOT}/data/enroll_close
test_dir=${VOX_ROOT}/data/test_close
python3 -u data/label.py --source_path ${train_dir}/utt2spk --dest_path data/train/spk.npz
python3 -u data/label.py --source_path ${test_dir}/utt2spk --dest_path data/test/spk.npz
echo

echo data_prepare all DONE!
