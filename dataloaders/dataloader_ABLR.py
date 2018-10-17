import sys
sys.path.append('..')
import json
import pickle as pkl
import numpy as np
import os
import nltk
import h5py
import random
from gensim.models import KeyedVectors

def load_file(filename):
    with open(filename,'rb') as fr:
        return pkl.load(fr)

def load_json(filename):
    with open(filename) as fr:
        return json.load(fr)



class Loader(object):
    def __init__(self, params, key_file, word2vec, is_training=False ):
        #general
        self.params = params
        self.is_training = is_training
        self.feature_path = params['feature_path']
        self.feature_path_tsn = params['feature_path_tsn']

        self.max_batch_size = params['batch_size']

        # dataset
        self.word2vec = word2vec
        self.key_file = load_json(key_file)
        if self.is_training == True:
            self.key_file = self.key_file[:12800]
        else:
            self.key_file = self.key_file[:1280]

        self.dataset_size = len(self.key_file)
        self.data_index = list(range(len(self.key_file)))
        self.next_idx = 0
        if self.is_training:
            random.shuffle(self.data_index)


        # frame / question
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']




    def reset(self):
        self.next_idx = 0
        if self.is_training:
            random.shuffle(self.data_index)


    def generate(self):

        while True:

            batch_size = min(self.max_batch_size, self.dataset_size - self.next_idx)
            if batch_size <= 0:
                yield None, None, None, None, None, None

            frame_vecs = np.zeros((batch_size,self.max_frames, self.input_video_dim), dtype=np.float32)
            frame_n = np.zeros((batch_size), dtype=np.int32)
            ques_vecs = np.zeros((batch_size,self.max_words, self.input_ques_dim), dtype=np.float32)
            ques_n = np.zeros((batch_size), dtype=np.int32)
            labels = np.zeros((batch_size,self.max_frames), dtype=np.float32)
            gt_windows = np.zeros((batch_size,2), dtype=np.float32)

            for i in range(batch_size):
                index = self.data_index[self.next_idx + i]
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

                inds = np.floor(np.arange(0, len(feats)-0.0000001, len(feats) / self.max_frames)).astype(int)
                frames = feats[inds, :]
                frames = np.vstack(frames)
                frame_vecs[i, :self.max_frames, :] = frames[:self.max_frames, :]
                frame_n[i] = self.max_frames

                frame_per_sec = self.max_frames/duration
                start_frame = round(frame_per_sec * timestamps[0])
                end_frame = round(frame_per_sec * timestamps[1]) - 1
                gt_windows[i,:] = [start_frame, end_frame]

                for j in range(self.max_frames):
                    if j >= start_frame and j <= end_frame:
                        labels[i,j] = 1

                    else:
                        labels[i,j] = 0

                # question
                stopwords = ['.', '?', ',', '']
                sent = nltk.word_tokenize(sent)
                ques = [word.lower() for word in sent if word not in stopwords]
                ques = [self.word2vec[word] for word in ques if word in self.word2vec]
                ques_feats = np.stack(ques, axis=0)
                ques_n[i] = min(len(ques), self.max_words)
                ques_vecs[i,:ques_n[i], :] = ques_feats[:ques_n[i], :]

            self.next_idx += batch_size
            yield frame_vecs, frame_n, ques_vecs, ques_n, labels, gt_windows



if __name__ == '__main__':

    config_file = '../configs/config_ABLR.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    word2vec = KeyedVectors.load_word2vec_format(config["word2vec"], binary=True)

    train_dataset = Loader(config, config['train_data'], word2vec, is_training = True)

    for frame_vecs, frame_n, ques_vecs, ques_n, labels, gt_windows in train_dataset.generate():
        print(frame_vecs)
        print(frame_vecs.shape)
        print(frame_n)
        print(frame_n.shape)
        print(ques_vecs)
        print(ques_vecs.shape)
        print(ques_n)
        print(ques_n.shape)
        print(labels)
        print(labels.shape)
        print(gt_windows)
        print(gt_windows.shape)


