import sys
sys.path.append('..')
import json
import pickle as pkl
import numpy as np
import random
import os
import nltk
import h5py
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader
from tools.criteria import calculate_IoU, calculate_nIoL

def load_file(filename):
    with open(filename,'rb') as fr:
        return pkl.load(fr)

def load_json(filename):
    with open(filename) as fr:
        return json.load(fr)



class Loader(Dataset):
    def __init__(self, params, key_file, ques_file, word2vec, flag=False ):
        #general
        self.params = params
        self.is_training = flag
        self.feature_path = params['feature_path']
        self.feature_path_tsn = params['feature_path_tsn']

        self.max_batch_size = params['batch_size']

        # dataset
        self.word2vec = word2vec
        self.key_file = load_json(key_file)
        self.ques_file = np.load(ques_file)
        self.dataset_size = len(self.key_file)


        # frame / question
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']



    def __getitem__(self, index):

        frame_vecs = np.zeros((self.max_frames, self.input_video_dim), dtype=np.float32)

        ques_vecs = self.ques_file[index]
        keys = self.key_file[index]
        vid, duration, timestamps, sent = keys[0], keys[1], keys[2], keys[3]

        # video
        if self.params['is_origin_dataset']:
            if not os.path.exists(self.feature_path + '/%s.h5' % vid):
                print('the video is not exist:', vid)
            with h5py.File(self.feature_path + '/%s.h5' % vid, 'r') as fr:
                feats = np.asarray(fr['feature'])
        else:
            vid = vid[2:]
            while not os.path.exists(self.feature_path_tsn + '/feat/%s.h5' % vid):
                keys = self.key_file[0]
                vid, duration, timestamps, sent = keys[0], keys[1], keys[2], keys[3]
                vid = vid[2:]

            with h5py.File(self.feature_path_tsn + '/feat/%s.h5' % vid, 'r') as hf:
                fg = np.asarray(hf['fg'])
                bg = np.asarray(hf['bg'])
                feat = np.hstack([fg, bg])
            with h5py.File(self.feature_path_tsn + '/flow/%s.h5' % vid, 'r') as hf:
                fg2 = np.asarray(hf['fg'])
                bg2 = np.asarray(hf['bg'])
                feat2 = np.hstack([fg2, bg2])
            feats = feat + feat2


        inds = np.floor(np.arange(0, len(feats), len(feats) / self.max_frames)).astype(int)
        frames = feats[inds, :]
        frames = np.vstack(frames)
        frame_vecs[:self.max_frames, :] = frames[:self.max_frames, :]
        frame_n = np.array(len(frame_vecs),dtype=np.int32)

        # [32,64,128,256] / [8,16,32,64]
        frame_per_sec = self.max_frames/duration
        start_frame = round(frame_per_sec * timestamps[0])
        end_frame = round(frame_per_sec * timestamps[1]) - 1
        gt_windows = np.array([start_frame, end_frame], dtype=np.float32)

        widths = np.array([32, 64, 128, 256])
        nums = (self.max_frames-widths*0.75)/(widths*0.25)
        nums = nums.astype(np.int) # [45,21,9,3]
        labels = [[0]*num for num in nums]
        windows = list()
        cur_best = -2
        best_window = [0, 0]
        best_pos = [0, 31]
        for i in range(len(widths)):
            num = nums[i]
            width = widths[i]
            step = widths[i]/4
            for j in range(num):
                start = j*step
                end = start + width - 1
                windows.append([start,end])
                iou = calculate_IoU([start_frame,end_frame],[start,end])
                niol = calculate_nIoL([start_frame,end_frame],[start,end])
                # if iou >= 0.5 and niol <= 0.2:
                if iou - niol >= cur_best:
                    cur_best = iou - niol
                    best_window = [i,j]
                    best_pos = [start, end]

        labels[best_window[0]][best_window[1]] = 1
        labels = np.hstack([labels[0],labels[1],labels[2],labels[3]]).astype(np.float32)
        regs = np.array([start_frame - best_pos[0], end_frame - best_pos[1]],dtype=np.float32)
        idxs = np.array(np.where((labels > 0)),dtype=np.int64)
        windows = np.array(windows, dtype=np.float32)

        return frame_vecs, frame_n, ques_vecs, labels, regs, idxs, windows, gt_windows



    def __len__(self):
        if self.is_training:
            return self.dataset_size
        else:
            return 1280

if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    word2vec = KeyedVectors.load_word2vec_format(config["word2vec"], binary=True)

    train_dataset = Loader(config, config['train_data'], word2vec)


    # Data loader (this provides queues and threads in a very simple way).
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    # Mini-batch images and labels.

    frame_vecs, frame_n, ques_vecs, ques_n, labels, regs, idxs, windows, gt_windows = data_iter.next()
    print(frame_vecs.dtype)
    print(frame_n.dtype)
    print(ques_vecs.dtype)
    print(ques_n.dtype)
    print(labels.dtype)
    print(regs.dtype)
    print(idxs.dtype)
    print(windows)
    print(gt_windows)

