import sys
sys.path.append('..')
import json
from dataloaders.dataloader_rnn import Loader
from models.model_rnn_GAN import Model
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




        # pre-train initialization
        pre_global_step = tf.get_variable('pre_global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        pre_learning_rates = tf.train.exponential_decay(self.params['learning_rate'], pre_global_step, decay_steps=self.params['lr_decay_n_iters'], decay_rate=self.params['lr_decay_rate'], staircase=True)
        pre_optimizer = tf.train.AdamOptimizer(pre_learning_rates)
        self.pre_train_proc = pre_optimizer.minimize(self.model.G_pre_loss, global_step=pre_global_step)

        # generator initialization
        self.g_train_proc = pre_optimizer.minimize(self.model.G_loss,global_step=pre_global_step,var_list=self.model.G_variables)

        # discriminator initialization
        d_global_step = tf.get_variable('d_global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        d_learning_rates = tf.train.exponential_decay(self.params['d_learning_rate'], d_global_step, decay_steps=self.params['d_lr_decay_n_iters'], decay_rate=self.params['d_lr_decay_rate'], staircase=True)
        d_optimizer = tf.train.AdamOptimizer(d_learning_rates)
        d_grads_and_vars = d_optimizer.compute_gradients(self.model.D_loss, self.model.D_variables, aggregation_method=2)
        self.d_train_proc = d_optimizer.apply_gradients(d_grads_and_vars, global_step=d_global_step)

        # pointer initialization
        pn_global_step = tf.get_variable('pn_global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        pn_learning_rates = tf.train.exponential_decay(self.params['pn_learning_rate'], pn_global_step,
                                                       decay_steps=self.params['pn_lr_decay_n_iters'],
                                                       decay_rate=self.params['pn_lr_decay_rate'], staircase=True)
        pn_optimizer = tf.train.AdamOptimizer(pn_learning_rates)
        pn_grads_and_vars = pn_optimizer.compute_gradients(self.model.pn_loss, self.model.pn_variables,
                                                           aggregation_method=2)
        self.pn_train_proc = pn_optimizer.apply_gradients(pn_grads_and_vars, global_step=pn_global_step)


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


        # G_pretrain
        for i_epoch in range(self.params['g_pretrain_epoch']):
            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch,self.pre_train_proc,self.model.G_pre_loss)
            t_end = time.time()
            print('G Pretrain Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch %2 == 0:
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.val_loader)
                print('=================================')

        t_begin = time.time()
        avg_batch_loss = self.train_one_epoch(0, self.pn_train_proc, self.model.pn_loss)
        t_end = time.time()
        print('PN Pretrain Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (
                                                        0, avg_batch_loss, t_end - t_begin))

        # D_pretrain
        for i_epoch in range(self.params['g_pretrain_epoch']):
            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch, self.d_train_proc, self.model.D_loss)
            t_end = time.time()
            print('D Pretrain Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

        # iter train
        for i_epoch in range(self.params['max_epoches']):
            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch, self.g_train_proc, self.model.G_loss)
            t_end = time.time()
            print('Epoch %d of G ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch % 2 == 0:
                t_begin = time.time()
                avg_batch_loss = self.train_one_epoch(i_epoch, self.pn_train_proc, self.model.pn_loss)
                t_end = time.time()
                print('PN Pretrain Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (
                    i_epoch, avg_batch_loss, t_end - t_begin))


            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch, self.d_train_proc, self.model.D_loss)
            t_end = time.time()
            print('Epoch %d of D ends. Average loss %.3f. %.3f seconds/epoch' % (
            i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch % self.params['evaluate_interval'] == 0 and i_epoch != 0:
                print('=================================')
                print('Overall evaluation')
                print('=================================')
                # print('train set evaluation')
                # train_acc = self.evaluate(self.train_loader)
                # print('=================================')
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
            for i in range(3):
                t_begin = time.time()
                avg_batch_loss = self.train_one_epoch(i, self.pn_train_proc, self.model.pn_loss)
                t_end = time.time()
                print('Post PN Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i, avg_batch_loss, t_end - t_begin))

                self.evaluate(self.test_loader)
        else:
            print('ERROR: No checkpoint available!')




    def train_one_epoch(self, i_epoch, train_proc, loss):

        loss_sum = 0
        pure_loss_sum = 0
        display_loss_sum = 0
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
            batch_data[self.model.gt_windows] = gt_windows
            batch_data[self.model.is_training] = True
            batch_data[self.model.batch_size] = len(frame_vecs)

            # Forward pass
            _, batch_loss, pure_loss = self.sess.run(
                                        [train_proc, loss, self.model.G_pre_loss], feed_dict=batch_data)


            i_batch += 1
            loss_sum += batch_loss
            pure_loss_sum += pure_loss
            display_loss_sum += batch_loss

            if i_batch % self.params['display_batch_interval'] == 0:
                t2 = time.time()
                print('Epoch %d, Batch %d, G_loss = %.4f, %.3f seconds/batch' % ( i_epoch, i_batch, display_loss_sum/self.params['display_batch_interval'] ,
                    (t2 - t1) / self.params['display_batch_interval']))
                print('Pure loss:', pure_loss_sum/self.params['display_batch_interval'])
                display_loss_sum = 0
                pure_loss_sum = 0
                t1 = t2

        avg_batch_loss = loss_sum / i_batch

        return avg_batch_loss



    def evaluate(self, data_loader):

        # top1,top5,top10
        data_loader.reset()
        IoU_thresh = [0.1, 0.3, 0.5, 0.7]
        all_correct_num_topn_IoU = np.zeros(shape=[1,4],dtype=np.float32)
        all_iou = 0
        all_pn_correct_num_topn_IoU = np.zeros(shape=[1,4],dtype=np.float32)
        all_pn_iou = 0
        all_retrievd = 0.0
        i_batch = 0
        loss_sum = 0
        pn_loss_sum = 0
        pure_loss_sum = 0
        d_loss_sum = 0


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
            batch_data[self.model.gt_windows] = gt_windows
            batch_data[self.model.is_training] = False
            batch_data[self.model.batch_size] = batch_size

            # Forward pass
            pn_loss, d_loss, pure_loss, batch_loss, frame_score, predict_start_end = self.sess.run(
                                        [self.model.pn_loss,self.model.D_loss, self.model.G_pre_loss, self.model.G_loss, self.model.frame_score, self.model.predict_start_end], feed_dict=batch_data)



            for i in range(batch_size):

                predict_score, predict_windows = self.propose_field(frame_score, batch_size, i_batch, i, gt_windows)
                propose_result = self.calculate_IoU(predict_windows[0], gt_windows[i])
                all_iou += propose_result
                for i in range(len(IoU_thresh)):
                    if propose_result >= IoU_thresh[i]:
                        all_correct_num_topn_IoU[0][i] += 1.0

                pn_result = self.calculate_IoU(predict_start_end[i], gt_windows[i])
                all_pn_iou += pn_result
                for i in range(len(IoU_thresh)):
                    if pn_result >= IoU_thresh[i]:
                        all_pn_correct_num_topn_IoU[0][i] += 1.0

            all_retrievd += batch_size
            i_batch += 1
            loss_sum += batch_loss
            pure_loss_sum += pure_loss
            d_loss_sum += d_loss
            pn_loss_sum += pn_loss


            if i_batch % 100 == 0:
                print('Batch %d, G_loss = %.4f, G_pure_loss = %.4f, D_loss = %.4f, PN_loss= %.4f' %
                      (i_batch, loss_sum / i_batch, pure_loss_sum/ i_batch, d_loss_sum/i_batch, pn_loss_sum/i_batch ))


        avg_correct_num_topn_IoU = all_correct_num_topn_IoU / all_retrievd
        avg_pn_correct_num_topn_IoU = all_pn_correct_num_topn_IoU / all_retrievd

        print('=================================')
        print('propose:',avg_correct_num_topn_IoU, all_iou/ all_retrievd)
        print('pn:',avg_pn_correct_num_topn_IoU, all_pn_iou/ all_retrievd)
        print('=================================')

        acc = avg_correct_num_topn_IoU[0,2]
        return acc

    def calculate_IoU(self, i0, i1):
        union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
        inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
        iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
        return iou


    def propose_field(self, frame_score, batch_size, i_batch, i, gt_windows):

        frame_pred = frame_score[i]
        frame_pred = (frame_pred - np.mean(frame_pred)) / np.std(frame_pred)
        scale = max(max(frame_pred), -min(frame_pred)) / 0.5
        frame_pred = frame_pred / (scale + 1e-3) + 0.5
        frame_pred_in = np.log(frame_pred)
        frame_pred_out = np.log(1 - frame_pred)
        candidate_num = 1
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

        # if i_batch % 100 == 0 and i_batch % batch_size == i:
        #     print(gt_windows[i])
        #     print(frame_pred)

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

        # if i_batch % 100 == 0 and i_batch % batch_size == i:
        #     print(predict_windows)

        return predict_score, predict_windows


if __name__ == '__main__':

    config_file = '../configs/config_rnn_GAN.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    trainer = Trainer(config)

    trainer.train()
