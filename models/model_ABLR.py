import sys
sys.path.append('..')
import json
import tensorflow as tf

import tools.layers as layers

def matmul(ts, mat):
    ts_s1 = int(ts.get_shape()[-2])
    ts_s2 = int(ts.get_shape()[-1])
    mat_s1 = int(mat.get_shape()[-1])
    reshaped = tf.reshape(ts, [-1, ts_s2])
    matmul = tf.matmul(reshaped, mat)
    output = tf.reshape(matmul, [-1, ts_s1, mat_s1])
    return output

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
        self.attention_size = params['attention_size']

        self.build_model()

    def co_attention(self,Z,g,scope):
        size_ = Z.get_shape().as_list()[1]
        with tf.variable_scope(scope):
            U_z = tf.Variable(tf.random_uniform([2*self.hidden_size,self.attention_size]),name = 'U_z')
            U_g = tf.Variable(tf.random_uniform([2*self.hidden_size,self.attention_size]),name = 'U_g')
            b_a = tf.Variable(tf.random_uniform([self.attention_size]),name = 'b_a')
            u_a = tf.Variable(tf.random_uniform([self.attention_size,1]),name = 'u_a')
            H = tf.tanh(matmul(Z,U_z)+tf.tile(tf.expand_dims(tf.matmul(g, U_g),1),[1,size_,1])+b_a)
            a_z = tf.nn.softmax(matmul(H,u_a))
            z = tf.squeeze(tf.matmul(tf.transpose(Z,[0,2,1]),a_z),axis=-1)
        return tf.squeeze(a_z,axis=-1),z # a_z (batch_size,size_,1) z (batch_size,attention_size)

    def build_model(self):
        # input layer (batch_size, n_steps, input_dim)
        self.ques_vecs = tf.placeholder(tf.float32, [None, self.max_words, self.input_ques_dim])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.frame_vecs = tf.placeholder(tf.float32, [None, self.max_frames, self.input_video_dim])
        self.frame_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)
        self.gt_windows = tf.placeholder(tf.float32, [None, 2])
        self.gt_predict = tf.placeholder(tf.float32, [None, self.max_frames])

        self.frame_mask = tf.sequence_mask(self.frame_len, maxlen=self.max_frames)
        self.ques_mask = tf.sequence_mask(self.ques_len, maxlen=self.max_words)



        with tf.variable_scope("Embedding_Encoder_Layer"):
            input_frame_vecs = tf.contrib.layers.dropout(self.frame_vecs, self.dropout, is_training=self.is_training)
            frame_embedding, _ = layers.dynamic_origin_bilstm_layer(input_frame_vecs, self.hidden_size, 'frame_embedding',input_len=self.frame_len)
            frame_embedding = tf.contrib.layers.dropout(frame_embedding, self.dropout, is_training=self.is_training)
            input_ques_vecs = tf.contrib.layers.dropout(self.ques_vecs, self.dropout, is_training=self.is_training)
            ques_embedding, ques_states = layers.dynamic_origin_bilstm_layer(input_ques_vecs, self.hidden_size, 'ques_embedding',input_len=self.ques_len)
            ques_embedding = tf.contrib.layers.dropout(ques_embedding, self.dropout, is_training=self.is_training)


        with tf.variable_scope("Co-Attention-Interaction"):
            mean_ques = tf.reduce_mean(ques_embedding,axis=1)
            _, embedded_1 = self.co_attention(frame_embedding,mean_ques,'embedded_1')
            _, embedded_2 = self.co_attention(ques_embedding,embedded_1,'embedded_2')
            self.a_v, embedded_3 = self.co_attention(frame_embedding,embedded_2,'embedded_3')


        with tf.variable_scope("Output_Layer"):
            self.predicted_position = tf.layers.dense(self.a_v,2,tf.nn.relu,name='predicted_position',kernel_initializer=tf.random_normal_initializer(),trainable=True)
            L_reg = tf.reduce_sum(tf.abs(tf.subtract(self.predicted_position,self.gt_windows)))
            L_cal = -tf.reduce_sum(tf.reduce_sum(tf.multiply(self.gt_predict,tf.log(self.a_v)),axis=-1) / tf.reduce_sum(self.gt_predict,axis=-1))
            avg_logit_loss = L_reg + 5 * L_cal

            variables = tf.trainable_variables()
            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
            self.test_loss = avg_logit_loss
            self.loss = avg_logit_loss + self.regularization_beta * regularization_cost

if __name__ == '__main__':

    config_file = '../configs/config_ABLR.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    model = Model(config)


