import sys
sys.path.append('..')
import json
from dataloaders.dataloader_TALL import Loader
from models.model_TALL import Model
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
        self.train_proc = self.optimizer.minimize(self.model.loss, global_step=global_step)
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
        valid_acc = 0
        print('=================================')
        print('Model Params:')
        print(self.params)
        print('=================================')

        # self.evaluate(self.val_loader)

        for i_epoch in range(self.params['max_epoches']):
            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch)
            t_end = time.time()
            print('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch % self.params['evaluate_interval'] == 0:
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
        for frame_vecs, ques_vecs, ques_n, gt_windows in self.train_loader.generate_train():
            if frame_vecs is None:
                break

            batch_data = dict()
            batch_data[self.model.frame_vecs] = frame_vecs
            batch_data[self.model.ques_vecs] = ques_vecs
            batch_data[self.model.ques_len] = ques_n
            batch_data[self.model.gt_windows] = gt_windows
            batch_data[self.model.is_training] = True

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
        t1 = time.time()
        # IoU_thresh = [0.5, 0.7]
        # top1,top5,top10
        data_loader.reset()
        all_correct_num_topn_IoU = np.zeros(shape=[2,4],dtype=np.float32)
        all_retrievd = 0.0
        i_batch = 0

        for frame_vecs, ques_vecs, ques_n, gt_windows in data_loader.generate_test():
            if frame_vecs is None:
                break

            batch_size = len(frame_vecs)
            for i in range(self.params['batch_size']):
                sim_score_mat = np.zeros([int(1+(self.params['max_frames'] -self.params['interval'])/ self.params['overlap']),3])
                for j in range(len(sim_score_mat)):
                    batch_data = dict()
                    batch_data[self.model.frame_vecs_test] = [frame_vecs[i][j]]
                    batch_data[self.model.ques_vecs_test] = [ques_vecs[i]]
                    batch_data[self.model.ques_len_test] = [ques_n[i]]
                    batch_data[self.model.is_training] = False
                    # print(batch_data)
                    # Forward pass
                    outputs = self.sess.run([self.model.sim_score_mat_test], feed_dict=batch_data)
                    sim_score_mat[j, :] = outputs[0]
                starts = np.array(range(0,self.params['max_frames'] - self.params["interval"]+1,self.params["overlap"]))
                ends = starts + self.params["interval"]
                sim_score_mat[:,1] = starts + sim_score_mat[:,1]
                sim_score_mat[:, 2] = ends + sim_score_mat[:, 2]
                predict_score = sim_score_mat[:,0]
                predict_windows = sim_score_mat[:,1:]
                result = criteria.compute_IoU_recall(predict_score, predict_windows, gt_windows[i])

                all_correct_num_topn_IoU += result
            all_retrievd += batch_size
            i_batch += 1

            # if i_batch % 10 == 0:
            #     print('Batch %d' % (i_batch))
        cost_time = time.time() - t1
        print("cost_time",cost_time)
        avg_correct_num_topn_IoU = all_correct_num_topn_IoU / all_retrievd
        print('=================================')
        print(avg_correct_num_topn_IoU)
        print('=================================')

        acc = avg_correct_num_topn_IoU[0,3]
        return acc

if __name__ == '__main__':

    config_file = '../configs/config_TALL.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    trainer = Trainer(config)

    trainer.train()
