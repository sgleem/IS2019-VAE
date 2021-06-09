# -*- coding: utf-8 -*-

# Author: Yang Zhang
# Mail: zyziszy@foxmail.com
# Apache 2.0.

import os
import numpy as np
# import tensorflow as tf  # tf-gpu 1.8
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from model.vae import *
from model.model_utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
'''flags'''

tf.compat.v1.flags.DEFINE_integer('epoch', 50, 'epoch num')

tf.compat.v1.flags.DEFINE_integer('batch_size', 200, 'batch size')

tf.compat.v1.flags.DEFINE_integer('n_hidden', 1800, 'dim of hidden')

tf.compat.v1.flags.DEFINE_integer('z_dim', 200, 'dim of z')

tf.compat.v1.flags.DEFINE_float('learn_rate', 0.00001, 'learn rate')

tf.compat.v1.flags.DEFINE_float('beta1', 0.5, 'beta1 for AdamOptimizer')

tf.compat.v1.flags.DEFINE_float('KL_weigth', 0.04, 'KL_weigth')

tf.compat.v1.flags.DEFINE_float('cohesive_weight', 0., 'cohesive loss')

tf.compat.v1.flags.DEFINE_string('dataset_path', './data/voxceleb_combined_200000/xvector.npz',
                           'x vector dataset path (npz format)')

tf.compat.v1.flags.DEFINE_string('spk_path', './data/voxceleb_combined_200000/spk.npz',
                           'utt2spk label dataset path (npz format)')

tf.compat.v1.flags.DEFINE_integer('is_training', 1, 'Training/Testing.')

params = tf.compat.v1.flags.FLAGS  # store flag

'''model's log and checkpoints paths'''
experiment_dir = os.path.join('experiments', 'z' + \
    str(params.z_dim)+'_h' + str(params.n_hidden) + \
    '_kl'+str(params.KL_weigth)+'_c'+str(params.cohesive_weight))

experiment_dir = os.path.dirname(os.path.abspath(__file__))+experiment_dir
checkpoint_dir = os.path.join(experiment_dir,'checkpoint')
log_dir = os.path.join(experiment_dir,'train_log')
print('model/checkpoint/logs will save in {}.'.format(experiment_dir))


'''build the model and train'''
with tf.compat.v1.Session() as sess:
    vae_model = VAE(
        sess=sess,
        epoch=params.epoch,
        batch_size=params.batch_size,
        z_dim=params.z_dim,
        dataset_path=params.dataset_path,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        n_hidden=params.n_hidden,
        KL_weigth=params.KL_weigth,
        cohesive_weight=params.cohesive_weight,
        learning_rate=params.learn_rate,
        beta1=params.beta1,
        spk_path=params.spk_path
    )
    if params.is_training:
        vae_model.train()
        print('model / checkpoint / logs will save in {}.'.format(experiment_dir))

    else:
        paths = ["./data/train/xvector", "./data/test/xvector"]
        for path in paths:
            if os.path.exists(path+'.ark') == True:
                os.remove(path+'.ark')
                print('delete {}.ark'.format(path))

        for path in paths:
            # load data
            vector = np.load(path+'.npz')['vector']
            labels = np.load(path+'.npz')['utt']
            # vector = vector[:int(len(vector)*0.5)]
            # predict
            predict_mu = vae_model.predict(vector)
            print(path)
            print(predict_mu.shape)
            # get_skew_and_kurt(predict_mu)
            with open(path+'_vae.ark', 'w') as f:
                for i in range(predict_mu.shape[0]):
                    f.write(str(labels[i]))
                    f.write('  [ ')
                    for j in predict_mu[i]:
                        f.write(str(j))
                        f.write(' ')
                    f.write(']')
                    f.write('\n')
            print('{}.ark is done!'.format(path))
        print('\nall done!')

print('done')
