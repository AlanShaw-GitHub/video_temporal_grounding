import sys
sys.path.append('..')
import json
import tensorflow as tf
from tools.main_layers import *




class Model(object):

    def __init__(self, params):

        self.params = params
        self.regularization_beta = params['regularization_beta']
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']
        self.hidden_size = params['hidden_size']
        self.num_heads = params['num_heads']


        self.build_model()


    def build_model(self):
        # input layer (batch_size, n_steps, input_dim)
        self.ques_vecs = tf.placeholder(tf.float32, [None, self.max_words, self.input_ques_dim])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.frame_vecs = tf.placeholder(tf.float32, [None, self.max_frames, self.input_video_dim])
        self.frame_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.is_training = tf.placeholder(tf.bool)
        self.start = tf.placeholder(tf.int32, [None, self.max_frames])
        self.end = tf.placeholder(tf.int32, [None, self.max_frames])
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

        self.frame_mask = tf.sequence_mask(self.frame_len, maxlen=self.max_frames)
        self.ques_mask = tf.sequence_mask(self.ques_len, maxlen=self.max_words)

        with tf.variable_scope("Input_Layer"):
            input_frame_vecs = linear_layer_3d(self.frame_vecs, self.hidden_size, scope_name='frame_input')
            input_ques_vecs = linear_layer_3d(self.ques_vecs, self.hidden_size, scope_name='ques_input')


        with tf.variable_scope("Embedding_Encoder_Layer"):
            video_encoder_features = residual_block(input_frame_vecs,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.frame_mask,
                num_filters = self.hidden_size,
                num_heads = self.num_heads,
                seq_len = self.frame_len,
                scope = "Encoder_Video_Residual_Block",
                bias = False,
                dropout = self.dropout)

            ques_encoder_features = residual_block(input_ques_vecs,
                num_blocks = 1,
                num_conv_layers = 3,
                kernel_size = 5,
                mask = self.ques_mask,
                num_filters = self.hidden_size,
                num_heads = self.num_heads,
                seq_len = self.ques_len,
                scope = "Encoder_Ques_Residual_Block",
                reuse = False, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):

            S = tf.matmul(video_encoder_features, ques_encoder_features, transpose_b=True)
            mask_q = tf.expand_dims(self.ques_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_v = tf.expand_dims(self.frame_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_v), dim = 1),(0,2,1))
            self.v2q = tf.matmul(S_, ques_encoder_features)
            self.q2v = tf.matmul(tf.matmul(S_, S_T), video_encoder_features)
            attention_outputs = [video_encoder_features, self.v2q, video_encoder_features * self.v2q, video_encoder_features * self.q2v]

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis = -1)
            self.enc = [conv(inputs, self.hidden_size, name = "input_projection")]
            for i in range(3):
                if i % 2 == 0:
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                        num_blocks = 1,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.frame_mask,
                        num_filters = self.hidden_size,
                        num_heads = self.num_heads,
                        seq_len = self.frame_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout)
                    )

        with tf.variable_scope("Output_Layer"):
            start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
            end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
            self.logits = [mask_logits(start_logits, mask = self.frame_mask),
                           mask_logits(end_logits, mask = self.frame_mask)]

            logits1, logits2 = [l for l in self.logits]

            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            self.predict_matrix = tf.matrix_band_part(outer, 0, self.max_words)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.start)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.end)
            self.loss = tf.reduce_mean(losses + losses2)

        variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
        self.loss += l2_loss




if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)



