import sys
sys.path.append('..')
import json
from dataloaders.dataloader_TALL import Loader
import tensorflow as tf
import tools.layers as layers



class Model(object):

    def __init__(self, params):

        self.params = params
        self.regularization_beta = params['regularization_beta']
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']
        self.hidden_size = params['hidden_size']
        self.dropout = params['dropout_prob']
        self.lambda_regression = params['lambda_regression']
        self.regularization_beta = params['regularization_beta']
        self.semantic_size = params['semantic_size']

        self.build_model()

    def cross_modal_comb(self, visual_feat, sentence_embed, batch_size):
        vv_feature = tf.reshape(tf.tile(visual_feat, [batch_size, 1]),
                                [batch_size, batch_size, self.semantic_size])
        ss_feature = tf.reshape(tf.tile(sentence_embed, [1, batch_size]), [batch_size, batch_size, self.semantic_size])
        concat_feature = tf.reshape(tf.concat([vv_feature, ss_feature],2),
                                    [batch_size, batch_size, self.semantic_size + self.semantic_size])

        mul_feature = tf.multiply(vv_feature, ss_feature)
        add_feature = tf.add(vv_feature, ss_feature)

        comb_feature = tf.reshape(tf.concat([mul_feature, add_feature, concat_feature],2),
                                  [batch_size, batch_size, self.semantic_size * 4])
        return comb_feature



    def build_model(self):
        # input layer (batch_size, n_steps, input_dim)
        self.ques_vecs = tf.placeholder(tf.float32, [None, self.max_words, self.input_ques_dim])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.frame_vecs = tf.placeholder(tf.float32, [None, 3*self.input_video_dim])
        self.gt_windows = tf.placeholder(tf.float32, [None, 2])
        self.ques_vecs_test = tf.placeholder(tf.float32, [1, self.max_words, self.input_ques_dim])
        self.ques_len_test = tf.placeholder(tf.int32, [1])
        self.frame_vecs_test = tf.placeholder(tf.float32, [1, 3 * self.input_video_dim])
        self.batch_size = self.params['batch_size']
        self.alpha = 1.0 / self.batch_size
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope("TALL_Model"):
            print("Building training network...............................\n")
            transformed_clip = tf.layers.dense(self.frame_vecs, units=self.semantic_size,name='frame_dense',kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.relu,trainable=True)
            transformed_clip_norm = tf.nn.l2_normalize(transformed_clip, axis=1)
            with tf.variable_scope('ques_lstm'):
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
                lstm_output, final_state = tf.nn.dynamic_rnn(cell,self.ques_vecs,self.ques_len,dtype=tf.float32)
            transformed_sentence = tf.layers.dense(tf.concat(final_state,axis=-1), units=self.semantic_size,name='ques_dense',kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.relu,trainable=True)
            transformed_sentence_norm = tf.nn.l2_normalize(transformed_sentence, axis=1)
            cross_modal_vec = self.cross_modal_comb(transformed_clip_norm, transformed_sentence_norm,
                                                          self.batch_size)
            _sim_score_mat = tf.layers.dense(cross_modal_vec, units=2*self.semantic_size,name='_final_dense',kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.relu,trainable=True)
            sim_score_mat = tf.layers.dense(cross_modal_vec, units=3,name='final_dense',kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.relu,trainable=True)

            self.sim_score_mat = tf.reshape(sim_score_mat, [self.batch_size, self.batch_size, 3])

            # -----testing part--------
            tf.get_variable_scope().reuse_variables()
            transformed_clip_test = tf.layers.dense(self.frame_vecs_test, units=self.semantic_size, name='frame_dense',kernel_initializer=tf.random_normal_initializer(),
                                               activation=tf.nn.relu, trainable=True)
            transformed_clip_norm_test = tf.nn.l2_normalize(transformed_clip_test, axis=1)
            with tf.variable_scope('ques_lstm'):
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
                lstm_output_test, final_state_test = tf.nn.dynamic_rnn(cell, self.ques_vecs_test, self.ques_len_test, dtype=tf.float32)
            transformed_sentence_test = tf.layers.dense(tf.concat(final_state_test, axis=-1), units=self.semantic_size,
                                                   name='ques_dense',kernel_initializer=tf.random_normal_initializer(), activation=tf.nn.relu, trainable=True)
            transformed_sentence_norm_test = tf.nn.l2_normalize(transformed_sentence_test, axis=1)
            cross_modal_vec_test = self.cross_modal_comb(transformed_clip_norm_test, transformed_sentence_norm_test,1)
            _sim_score_mat_test = tf.layers.dense(cross_modal_vec_test, units=2*self.semantic_size,name='_final_dense',kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.relu,trainable=True)
            sim_score_mat_test = tf.layers.dense(cross_modal_vec_test, units=3, name='final_dense',kernel_initializer=tf.random_normal_initializer(), activation=tf.nn.relu,
                                            trainable=True)
            self.sim_score_mat_test = tf.reshape(sim_score_mat_test, [3])

            with tf.variable_scope('compute_loss_reg'):
                sim_score_mat, p_reg_mat, l_reg_mat = tf.split(self.sim_score_mat, 3, -1)
                sim_score_mat = tf.reshape(sim_score_mat, [self.batch_size, self.batch_size])
                l_reg_mat = tf.reshape(l_reg_mat, [self.batch_size, self.batch_size])
                p_reg_mat = tf.reshape(p_reg_mat, [self.batch_size, self.batch_size])
                # unit matrix with -2
                I_2 = tf.diag(tf.constant(-2.0, shape=[self.batch_size]))
                all1 = tf.constant(1.0, shape=[self.batch_size, self.batch_size])
                mask_mat = tf.add(I_2, all1)
                # loss cls, not considering iou
                I = tf.diag(tf.constant(1.0, shape=[self.batch_size]))
                batch_para_mat = tf.constant(self.alpha, shape=[self.batch_size, self.batch_size])
                para_mat = tf.add(I, batch_para_mat)
                loss_mat = tf.log(tf.add(all1, tf.exp(tf.multiply(mask_mat, sim_score_mat))))
                loss_mat = tf.multiply(loss_mat, para_mat)
                self.loss_align = tf.reduce_mean(loss_mat)
                # regression loss
                l_reg_diag = tf.matmul(tf.multiply(l_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
                p_reg_diag = tf.matmul(tf.multiply(p_reg_mat, I), tf.constant(1.0, shape=[self.batch_size, 1]))
                self.offset_pred = tf.concat((p_reg_diag, l_reg_diag),1)
                self.loss_reg = tf.reduce_mean(tf.abs(tf.subtract(self.offset_pred, self.gt_windows)))

                self.loss = tf.add(tf.multiply(self.lambda_regression, self.loss_reg), self.loss_align)



if __name__ == '__main__':

    config_file = '../configs/config_TALL.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)
    model = Model(config)


