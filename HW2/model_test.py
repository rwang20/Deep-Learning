import tensorflow as tf
import numpy as np
import os
import sys
import json
import pandas as pd
import argparse
import random
import pickle

from colors import *
from tqdm import *
from datasetProcessing import DatasetTest
from utils import inv_sigmoid, linear_decay, dec_print_train, dec_print_val, dec_print_test
from model_s2s_att import S2VT
from subprocess import call

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.reset_default_graph()
  
test_mode = True
with_attention = True


learning_rate = 1e-4
num_epoches = 100
batch_size = 100 
num_display_steps = 20
num_saver_epoches = 10
save_dir = '/Users/rongwang/Box Sync/CPSC 8430 Deep Learning/HW2/save_models'
             # '/content/drive/My Drive/save_models'
log_dir = '/Users/rongwang/Box Sync/CPSC 8430 Deep Learning/HW2/logs'
            # '/content/drive/My Drive/log'
output_filename = '/Users/rongwang/Box Sync/CPSC 8430 Deep Learning/HW2/output.txt'
            # '/content/drive/My Drive/output.txt'
data_dir = '/Users/rongwang/Desktop/MLDS_hw2_1_data/'
            #  '/content/drive/MyDrive/MLDS_hw2_1_data'
test_dir = '/Users/rongwang/Desktop/MLDS_hw2_1_data/testing_data'
            #  '/content/drive/My Drive/MLDS_hw2_1_data/testing_data'


random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

n_inputs        = 4096
n_hidden        = 512
val_batch_size  = 100 
n_frames        = 80
max_caption_len = 50
forget_bias_red = 1.0
forget_bias_gre = 1.0
dropout_prob    = 0.5
n_attention     = n_hidden

special_tokens  = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
phases = {'train': 0, 'val': 1, 'test': 2}

def test_print(pred, idx2word, batch_size, id_batch):
    
    seq = []
    for i in range(0, batch_size):
        eos_pred = max_caption_len - 1
        for j in range(0, max_caption_len):
                if pred[i][j] == special_tokens['<EOS>']:
                    eos_pred = j
                    break
        pre = list( map (lambda x: idx2word[x] , pred[i][0:eos_pred])  )
        print(colors('\nid: ' + str(id_batch[i]) + '\nlen: ' + str(eos_pred) + '\nprediction: ' + 
                     str(pre),  fg='white', bg='green'))
        pre_no_eos = list( map (lambda x: idx2word[x] , pred[i][0:(eos_pred)]) )
        sen = ' '.join([w for w in pre_no_eos])
        seq.append(sen)
    return seq


def test():
    # data_dir no use! only for datasetTest
    datasetTest = DatasetTest(data_dir, test_dir, batch_size)
    datasetTest.build_test_data_obj_list()
    vocab_num = datasetTest.load_tokenizer()

    test_graph = tf.Graph()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with test_graph.as_default():
        feat = tf.placeholder(tf.float32, [None, n_frames, n_inputs], name='video_features')
        model = S2VT(vocab_num=vocab_num, with_attention=with_attention)
        logits, _, _ = model.build_model(feat, phase=phases['test'])
        dec_pred = model.inference(logits)

        model.set_saver(tf.train.Saver(max_to_keep=3))
    sess = tf.Session(graph=test_graph, config=gpu_config)

    
    ckpts_path = save_dir + "save_net.ckpt"
    ckpts_dir = os.path.dirname(ckpts_path)
    saver_path = ckpts_path
    print('model path: ' + saver_path)
    latest_checkpoint = tf.train.latest_checkpoint('/Users/rongwang/Box Sync/CPSC 8430 Deep Learning/HW2/save_models')
    
    model.saver.restore(sess, latest_checkpoint)
    print("Model Loaded: " + latest_checkpoint)

    txt = open(output_filename, 'w')

    num_steps = int( datasetTest.batch_max_size / batch_size)
    for i in range(0, num_steps):

        data_batch, id_batch = datasetTest.next_batch()
        p = sess.run(dec_pred, feed_dict={feat: data_batch})
        seq = dec_print_test(p, datasetTest.idx_to_word, batch_size, id_batch)

        for j in range(0, batch_size):
            txt.write(id_batch[j] + "," + seq[j] + "\n")

        print("Inference: " + str((i+1) * batch_size) + "/" + \
                str(datasetTest.batch_max_size) + ", done..." )
    
    print('\n\nTesting finished.')
    print('\n Save file: ' + output_filename)
    txt.close()

def main(_):
    test()

if __name__ == '__main__':
    tf.app.run(main=main)
    