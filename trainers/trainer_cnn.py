import sys
sys.path.append('..')
import json
from dataloaders.dataloader_rnn import Loader
from models.model_cnn import Model
import time
from gensim.models import KeyedVectors
import os
import numpy as np
import tools.criteria as criteria
import tensorflow as tf


class Trainer(object):
    def __init__(self, params):

        self.params = params
        self.word2vec = KeyedVectors.load_word2vec_format(params["word2vec"], binary=True)

        self.model = Model(params)

        self.train_loader = Loader(params, params['train_data'], self.word2vec, is_training=True)
        self.val_loader = Loader(params, params['val_data'], self.word2vec)
        self.test_loader = Loader(params, params['test_data'], self.word2vec)



        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rates = tf.train.exponential_decay(self.params['learning_rate'], global_step,
                                                    decay_steps=self.params['lr_decay_n_iters'],
                                                    decay_rate=self.params['lr_decay_rate'], staircase=True)




        self.optimizer = tf.train.AdamOptimizer(learning_rates)
        gradients = self.optimizer.compute_gradients(self.model.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_proc = self.optimizer.apply_gradients(capped_gradients, global_step)

        # self.train_proc = self.optimizer.minimize(self.model.loss, global_step=global_step)



        self.model_path = os.path.join(self.params['cache_dir'])
        if not os.path.exists(self.model_path):
            print('create path: ', self.model_path)
            os.makedirs(self.model_path)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        init_proc = tf.global_variables_initializer()
        self.sess.run(init_proc)

        self.model_saver = tf.train.Saver()
        self.last_checkpoint = None



    def train(self):
        print('Trainnning begins......')
        best_epoch_acc = 0
        best_epoch_id = 0

        print('=================================')
        print('Model Params:')
        print(self.params)
        print('=================================')

        self.evaluate(self.val_loader)

        for i_epoch in range(self.params['max_epoches']):
            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch)
            t_end = time.time()
            print('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch % self.params['evaluate_interval'] == 0 and i_epoch != 0:
                print('=================================')
                print('Overall evaluation')
                print('=================================')
                print('train set evaluation')
                train_acc = self.evaluate(self.train_loader)
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.val_loader)
                print('=================================')
                # print('test set evaluation')
                # test_acc = self.evaluate(self.test_loader)
                # print('=================================')
            else:
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.val_loader)
                print('=================================')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.last_checkpoint = self.model_saver.save(self.sess, self.model_path + timestamp)
            else:
                if i_epoch - best_epoch_id >= self.params['early_stopping']:
                    print('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

        print('=================================')
        print('Evaluating best model in file', self.last_checkpoint, '...')
        if self.last_checkpoint is not None:
            self.model_saver.restore(self.sess, self.last_checkpoint)
            self.evaluate(self.test_loader)
        else:
            print('ERROR: No checkpoint available!')




    def train_one_epoch(self, i_epoch):

        loss_sum = 0
        t1 = time.time()
        i_batch = 0

        self.train_loader.reset()
        for frame_vecs, frame_n, ques_vecs, ques_n, labels, gt_windows in self.train_loader.generate():

            if frame_vecs is None:
                break

            batch_data = dict()
            batch_data[self.model.frame_vecs] = frame_vecs
            batch_data[self.model.frame_len] = frame_n
            batch_data[self.model.ques_vecs] = ques_vecs
            batch_data[self.model.ques_len] = ques_n
            batch_data[self.model.gt_predict] = labels
            batch_data[self.model.is_training] = True
            batch_data[self.model.batch_size] = len(frame_vecs)

            # Forward pass
            _, batch_loss = self.sess.run(
                                        [self.train_proc, self.model.loss], feed_dict=batch_data)



            i_batch += 1
            loss_sum += batch_loss

            if i_batch % self.params['display_batch_interval'] == 0:
                t2 = time.time()
                print('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % ( i_epoch, i_batch, loss_sum / i_batch ,
                    (t2 - t1) / self.params['display_batch_interval']))
                t1 = t2

        avg_batch_loss = loss_sum / i_batch

        return avg_batch_loss



    def evaluate(self, data_loader):

        # IoU_thresh = [0.5, 0.7]
        # top1,top5
        data_loader.reset()
        all_correct_num_topn_IoU = np.zeros(shape=[2,2],dtype=np.float32)
        all_retrievd = 0.0
        i_batch = 0
        loss_sum = 0

        for frame_vecs, frame_n, ques_vecs, ques_n, labels, gt_windows in data_loader.generate():

            if frame_vecs is None:
                break

            batch_size = len(frame_vecs)
            batch_data = dict()
            batch_data[self.model.frame_vecs] = frame_vecs
            batch_data[self.model.frame_len] = frame_n
            batch_data[self.model.ques_vecs] = ques_vecs
            batch_data[self.model.ques_len] = ques_n
            batch_data[self.model.gt_predict] = labels
            batch_data[self.model.is_training] = False
            batch_data[self.model.batch_size] = batch_size

            # Forward pass
            batch_loss, frame_score = self.sess.run(
                                        [self.model.test_loss, self.model.frame_score], feed_dict=batch_data)



            for i in range(batch_size):


                predict_score, predict_windows = self.propose_field(frame_score, batch_size, i_batch, i, gt_windows)
                result = criteria.compute_IoU_recall(predict_score, predict_windows, gt_windows[i])
                all_correct_num_topn_IoU += result

            all_retrievd += batch_size
            i_batch += 1
            loss_sum += batch_loss


            if i_batch % 100 == 0:
                print('Batch %d, loss = %.4f' % (i_batch, loss_sum / i_batch))


        avg_correct_num_topn_IoU = all_correct_num_topn_IoU / all_retrievd
        print('=================================')
        print(avg_correct_num_topn_IoU)
        print('=================================')

        acc = avg_correct_num_topn_IoU[0,1]
        return acc


    def propose_field(self, frame_score, batch_size, i_batch, i, gt_windows):

        frame_pred = frame_score[i]
        frame_pred = (frame_pred - np.mean(frame_pred)) / np.std(frame_pred)
        scale = max(max(frame_pred), -min(frame_pred)) / 0.5
        frame_pred = frame_pred / (scale + 1e-3) + 0.5
        frame_pred_in = np.log(frame_pred)
        frame_pred_out = np.log(1 - frame_pred)
        candidate_num = 20
        start_end_matrix = np.zeros([self.params['max_frames'], self.params['max_frames']], dtype=np.float32)
        for start in range(self.params['max_frames']):
            for end in range(self.params['max_frames']):
                if start == end:
                    start_end_matrix[start, end] = frame_pred_in[start] + np.sum(frame_pred_out[:start]) + np.sum(
                        frame_pred_out[end + 1:])
                elif end > start:
                    start_end_matrix[start, end] = start_end_matrix[start, end - 1] + frame_pred_in[end] - \
                                                   frame_pred_out[end]
                else:
                    start_end_matrix[start, end] = -1e9

        if i_batch % 100 == 0 and i_batch % batch_size == i:
            print(gt_windows[i])
            print(frame_pred)

        predict_matrix_i = start_end_matrix

        predict_score = np.zeros([candidate_num], dtype=np.float32)
        predict_windows = np.zeros([candidate_num, 2], dtype=np.float32)

        for cond_i in range(candidate_num):
            max_v = np.max(predict_matrix_i)
            idxs = np.where(predict_matrix_i == max_v)
            start = idxs[0][0]
            end = idxs[1][0]

            predict_score[cond_i] = max_v
            predict_windows[cond_i, :] = [start, end]

            start_left = max(start - 10, 0)
            start_right = min(start + 10, self.params['max_frames'])
            end_left = max(end - 10, 0)
            end_right = min(end + 10, self.params['max_frames'])

            predict_matrix_i[start_left:start_right, end_left:end_right] = -1e10

        if i_batch % 100 == 0 and i_batch % batch_size == i:
            print(predict_windows)

        return predict_score, predict_windows


if __name__ == '__main__':

    config_file = '../configs/config_cnn.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    trainer = Trainer(config)

    trainer.train()
