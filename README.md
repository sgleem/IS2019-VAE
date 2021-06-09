# v-vector-tf

Tensorflow and kaldi implementation of our Interspeech2019 paper [VAE-based regularization for deep speaker embedding](https://github.com/zyzisyz/v-vector-tf/raw/master/paper.pdf)

**note: the repo is not the final release, I will clean up our experiemental code and update soon**

## Dependency

1. computer
2. Linux (centos 7)
3. conda (Python 3.6)
4. Tensorflow-gpu 1.8
5. kaldi-toolkit

## Datasets and X-vector

1. VoxCeleb
2. SITW
3. CSLT_SITW

## Steps

1. use kaldi to extract x-vector from uttrance and get `xvector.ark` files
2. covert the kaldi `xvector.ark` files to numpy binary data format (`xvector.ark` -> `xvector.npz`)
3. use tensorflow to train a VAE model, and get the V-vectors
4. use kaldi recipes to calculate EER (equal error rate)

## Usage

1. In Linux shell, type `bash data/data_prepare.sh`
2. In Anaconda Prompt, type 
```bash
python3 -u main.py --epoch 200 --batch_size 200 --n_hidden 1800 --learn_rate 0.00001 --beta1 0.5 --dataset_path .\\data\\train\\xvector.npz --spk_path .\\data\\train\\spk.npz --z_dim 200 --KL_weigth 0.03 --cohesive_weight 10 --is_training 1
```

3. In Linux shell, type `bash eval.sh`

4. Go to egs/voxceleb/v2

5. In run.sh, set stage=10

6. type `bash run.sh`

## Our result

SITW Dev. Core

|          |  Cosine  |   PCA    |   PLDA   |  L-PLDA  |  P-PLDA  |
| :------: | :------: | :------: | :------: | :------: | :------: |
| x-vector |  15.67   |  16.17   |   9.09   | **3.12** |   4.16   |
| a-vector |  16.10   |  16.48   |  11.21   |   4.24   |   5.01   |
| v-vector |  10.32   |   9.94   |   3.62   |   3.54   |   4.31   |
| c-vector | **9.05** | **8.55** | **3.50** |   3.31   | **3.85** |

Read the paper for more detail

## About

Licensed under the Apache License, Version 2.0, Copyright [zyzisyz](https://github.com/zyzisyz)

### Repo Author

Yang Zhang (zyziszy@foxmail.com)

### Contributors

- [@Lilt](http://166.111.134.19:8081/lilt/)
- [@fatejessie](https://github.com/fatejessie)
- [@xDarkLemon](https://github.com/xDarkLemon)
- [@AlanXiuxiu](https://github.com/AlanXiuxiu)
- @Z.K.
