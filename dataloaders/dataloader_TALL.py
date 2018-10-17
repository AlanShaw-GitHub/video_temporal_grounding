import sys
sys.path.append('..')
import json
import pickle as pkl
import numpy as np
import os
import nltk
import tools.criteria as criteria
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


    def generate_train(self):
        while True:
            batch_size = self.max_batch_size
            if self.dataset_size - self.next_idx <= self.max_batch_size:
                yield None, None, None, None

            frame_vecs = np.zeros((batch_size,3*self.input_video_dim), dtype=np.float32)
            ques_vecs = np.zeros((batch_size,self.max_words, self.input_ques_dim), dtype=np.float32)
            ques_n = np.zeros((batch_size), dtype=np.int32)
            gt_windows = np.zeros((batch_size,2), dtype=np.float32)
            tmp_batch_size = 0
            while True:
                self.next_idx += 1
                if self.next_idx >= self.dataset_size:
                    yield None, None, None, None
                index = self.data_index[self.next_idx]
                keys = self.key_file[index]
                vid, duration, timestamps, _sent = keys[0], keys[1], keys[2], keys[3]

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
                        vid, duration, timestamps, _sent = keys[0], keys[1], keys[2], keys[3]
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
                frame_per_sec = self.max_frames/duration
                start_frame = round(frame_per_sec * timestamps[0])
                end_frame = round(frame_per_sec * timestamps[1]) - 1
                seed = 0
                for i in range(0,self.max_frames - self.params["interval"]+1,self.params["overlap"]):
                    iou = criteria.calculate_IoU((i, i+self.params['interval']), (start_frame, end_frame))
                    if iou > 0.5:
                        nIoL = criteria.calculate_nIoL((start_frame, end_frame), (i, i+self.params['interval']))
                        if nIoL < 0.15:
                            #print(frames.shape)
                            #print(i+self.params['interval'])
                            middle = np.mean(frames[i:i+self.params['interval'],:], axis=0)
                            if i-self.params['overlap'] > 0:
                                before = np.mean(frames[i-self.params['overlap']:i+self.params['interval']-self.params['overlap'],:], axis=0)
                            else:
                                before = np.zeros([self.input_video_dim])
                            if i+self.params['interval']+self.params['overlap'] < self.max_frames:
                                after = np.mean(frames[i + self.params['overlap']:i+self.params['interval']+self.params['overlap'],:], axis=0)
                            else:
                                after = np.zeros([self.input_video_dim])
                            frame_vecs[tmp_batch_size,:] = np.hstack([before,middle,after])
                            # print('before')
                            # print(before)
                            # print('middle')
                            # print(middle)
                            # print('after')
                            # print(after)
                            gt_windows[tmp_batch_size, :] = [start_frame-i, end_frame-(i+self.params['interval'])]

                            # question
                            stopwords = ['.', '?', ',', '']
                            sent = nltk.word_tokenize(_sent)
                            ques = [word.lower() for word in sent if word not in stopwords]
                            ques = [self.word2vec[word] for word in ques if word in self.word2vec]
                            ques_feats = np.stack(ques, axis=0)
                            ques_n[tmp_batch_size] = min(len(ques), self.max_words)
                            ques_vecs[tmp_batch_size,:ques_n[tmp_batch_size], :] = ques_feats[:ques_n[tmp_batch_size], :]
                            tmp_batch_size += 1
                            if tmp_batch_size == self.max_batch_size:
                                seed = 1
                                break
                if seed == 1:
                    break
            # print(self.next_idx)
            yield frame_vecs, ques_vecs, ques_n, gt_windows

    def generate_test(self):
        while True:
            batch_size = self.max_batch_size
            if self.dataset_size - self.next_idx <= self.max_batch_size:
                yield None, None, None, None

            frame_vecs = np.zeros((batch_size,int(1+(self.params['max_frames'] -self.params['interval'])/ self.params['overlap']),3*self.input_video_dim), dtype=np.float32)
            ques_vecs = np.zeros((batch_size,self.max_words, self.input_ques_dim), dtype=np.float32)
            ques_n = np.zeros((batch_size), dtype=np.int32)
            gt_windows = np.zeros((batch_size,2), dtype=np.float32)
            for i in range(batch_size):
                self.next_idx += 1
                index = self.data_index[self.next_idx]
                keys = self.key_file[index]
                vid, duration, timestamps, _sent = keys[0], keys[1], keys[2], keys[3]

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
                        vid, duration, timestamps, _sent = keys[0], keys[1], keys[2], keys[3]
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

                # print('frame_vecs.shape',frame_vecs.shape)
                frame_per_sec = self.max_frames/duration
                start_frame = round(frame_per_sec * timestamps[0])
                end_frame = round(frame_per_sec * timestamps[1]) - 1
                for j in range(0,self.max_frames - self.params["interval"]+1,self.params["overlap"]):
                    middle = np.mean(frames[j:j+self.params['interval'],:], axis=0)
                    if j-self.params['overlap'] > 0:
                        before = np.mean(frames[j-self.params['overlap']:j+self.params['interval']-self.params['overlap'],:], axis=0)
                    else:
                        before = np.zeros([self.input_video_dim])
                    if j+self.params['interval']+self.params['overlap'] < self.max_frames:
                        after = np.mean(frames[j + self.params['overlap']:j+self.params['interval']+self.params['overlap'],:], axis=0)
                    else:
                        after = np.zeros([self.input_video_dim])
                    frame_vecs[i,int(j / self.params['overlap']),:] = np.hstack([before,middle,after])

                gt_windows[i, :] = [start_frame, end_frame]

                # question
                stopwords = ['.', '?', ',', '']
                sent = nltk.word_tokenize(_sent)
                ques = [word.lower() for word in sent if word not in stopwords]
                ques = [self.word2vec[word] for word in ques if word in self.word2vec]
                ques_feats = np.stack(ques, axis=0)
                ques_n[i] = min(len(ques), self.max_words)
                ques_vecs[i,:ques_n[i], :] = ques_feats[:ques_n[i], :]
            yield frame_vecs, ques_vecs, ques_n, gt_windows

if __name__ == '__main__':

    config_file = '../configs/config_TALL.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    word2vec = KeyedVectors.load_word2vec_format(config["word2vec"], binary=True)

    train_dataset = Loader(config, config['train_data'], word2vec, is_training = True)
    average = 0
    counts = 0
    for frame_vecs, ques_vecs, ques_n, gt_windows in train_dataset.generate_test():
        print('--------------------------------')
        try:
            #print(frame_vecs)
            print(frame_vecs[0].shape)
            # print(frame_n)
            # print(frame_n.shape)
            # print(ques_vecs)
            print(ques_vecs[0].shape)
            #print(ques_n)
            # print(ques_n.shape)
            # print(labels)
            # print(labels.shape)
            print(gt_windows[0])
            # print(gt_windows.shape)
            counts +=1

        except:
            break
    # print(average)
    print(counts)
    # print(average / counts)


