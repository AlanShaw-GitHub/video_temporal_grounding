import sys
sys.path.append('..')
import json
import tensorflow as tf
import tools.layers as layers
from tools.modules import *


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
        self.regularization_beta = params['regularization_beta']


        self.build_model()


    def build_model(self):
        # input layer (batch_size, n_steps, input_dim)
        self.ques_vecs = tf.placeholder(tf.float32, [None, self.max_words, self.input_ques_dim])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.frame_vecs = tf.placeholder(tf.float32, [None, self.max_frames, self.input_video_dim])
        self.frame_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)
        self.gt_predict = tf.placeholder(tf.float32, [None, self.max_frames])

        self.ques = self.ques_vecs
        self.video = self.frame_vecs

        with tf.variable_scope("encoder"):
            ## Blocks
            for i in range(6):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.ques = multihead_attention(queries=self.ques,
                                                   keys=self.ques,
                                                   num_units=self.hidden_size,
                                                   num_heads=8,
                                                   dropout_rate=self.dropout,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope = "self_attention")

                    ### Feed Forward
                    self.ques = feedforward(self.ques, num_units=[4 * self.hidden_size, self.hidden_size])

        with tf.variable_scope("decoder"):
            ## Blocks
            for i in range(6):
                with tf.variable_scope("num_blocks_{}".format(i)):

                    ## Multihead Attention ( vanilla attention)
                    self.video = multihead_attention(queries=self.video,
                                                   keys=self.ques,
                                                   num_units=self.hidden_size,
                                                   num_heads=8,
                                                   dropout_rate=self.dropout,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")

                    ## Feed Forward
                    self.video = feedforward(self.video, num_units=[4 * self.hidden_size, self.hidden_size])

        with tf.variable_scope("Output_Layer"):

            logit_score = layers.linear_layer_3d(self.video, 1, scope_name='output_layer')
            logit_score = tf.squeeze(logit_score, 2)
            logit_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits= logit_score, labels=self.gt_predict)
            avg_logit_loss = tf.reduce_mean(tf.reduce_sum(logit_loss,1))

            variables = tf.trainable_variables()
            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
            self.test_loss = avg_logit_loss
            self.loss = avg_logit_loss + self.regularization_beta * regularization_cost
            self.frame_score = tf.nn.sigmoid(logit_score)




if __name__ == '__main__':

    config_file = '../configs/config_rnn.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)



