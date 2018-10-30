import sys
sys.path.append('..')
import json
import tensorflow as tf
import tools.layers as layers
import tools.conv_utils as conv_utils
import tools.transformer as transformer


class Model(object):

    def __init__(self, params):

        self.params = params
        self.regularization_beta = params['regularization_beta']
        self.max_frames = params['max_frames']
        self.input_video_dim = params['input_video_dim']
        self.max_words = params['max_words']
        self.input_ques_dim = params['input_ques_dim']
        self.hidden_size = params['hidden_size']
        self.regularization_beta = params['regularization_beta']
        self.num_heads = params["num_heads"]
        self.dropout = params["dropout_prob"]

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
        self.frame_mask = tf.sequence_mask(self.frame_len, maxlen=self.max_frames)
        self.ques_mask = tf.sequence_mask(self.ques_len, maxlen=self.max_words)



        with tf.variable_scope("Frame_Embedding_Encoder_Layer"):

            frame_next_layer = tf.contrib.layers.dropout(self.frame_vecs, self.dropout, is_training=self.is_training)
            frame_next_layer = conv_utils.linear_mapping(frame_next_layer, self.hidden_size,
                                                         dropout=self.dropout,
                                                         var_scope_name="linear_mapping_before_cnn")
            frame_next_layer = transformer.normalize(frame_next_layer)

            frame_next_layer += transformer.positional_encoding_v2(frame_next_layer,
                                            num_units=self.hidden_size,
                                            zero_pad=False,
                                            scale=False,
                                            scope="enc_pe")


            for i in range(3):
                with tf.variable_scope("stack_%s"%i):


                    frame_next_layer = conv_utils.conv_encoder_stack(frame_next_layer, [self.hidden_size,self.hidden_size,self.hidden_size], [3,3,3],
                                                          {'src': self.dropout,
                                                           'hid': self.dropout}, mode=self.is_training)

                    frame_next_layer = transformer.multihead_attention(queries=frame_next_layer,
                                                   keys=frame_next_layer,
                                                   num_units=self.hidden_size,
                                                   num_heads=4,
                                                   dropout_rate=1-self.dropout,
                                                   is_training=self.is_training,
                                                   causality=False)

                    frame_next_layer = transformer.feedforward(frame_next_layer, num_units=[2 * self.hidden_size, self.hidden_size],is_training=self.is_training)


            frame_embedding = tf.contrib.layers.dropout(frame_next_layer, self.dropout, is_training=self.is_training)

        with tf.variable_scope("Ques_Embedding_Encoder_Layer"):

            ques_next_layer = tf.contrib.layers.dropout(self.ques_vecs, self.dropout, is_training=self.is_training)


            ques_next_layer = conv_utils.linear_mapping(ques_next_layer, self.hidden_size,
                                                             dropout=self.dropout,
                                                             var_scope_name="linear_mapping_before_cnn")
            ques_next_layer = transformer.normalize(ques_next_layer)


            ques_next_layer += transformer.positional_encoding_v2(ques_next_layer,
                                                               num_units=self.hidden_size,
                                                               zero_pad=False,
                                                               scale=False,
                                                               scope="enc_pe")

            for i in range(1):
                with tf.variable_scope("stack_%s"%i):

                    ques_next_layer = conv_utils.conv_encoder_stack(ques_next_layer, [self.hidden_size,self.hidden_size], [3,3],
                                                              {'src': self.dropout,
                                                               'hid': self.dropout}, mode=self.is_training)
                    ques_next_layer = transformer.multihead_attention(queries=ques_next_layer,
                                                                       keys=ques_next_layer,
                                                                       num_units=self.hidden_size,
                                                                       num_heads=4,
                                                                       dropout_rate=1 - self.dropout,
                                                                       is_training=self.is_training,
                                                                       causality=False)

                    ques_next_layer = transformer.feedforward(ques_next_layer,
                                                               num_units=[2 * self.hidden_size, self.hidden_size],is_training=self.is_training)

            ques_embedding = tf.contrib.layers.dropout(ques_next_layer, self.dropout, is_training=self.is_training)




        with tf.variable_scope("Context_to_Query_Attention_Layer"):

            att_score = tf.matmul(frame_embedding, ques_embedding, transpose_b=True)  # M*N1*K  ** M*N2*K  --> M*N1*N2
            mask_q = tf.expand_dims(self.ques_mask, 1)
            S_ = tf.nn.softmax(layers.mask_logits(att_score, mask = mask_q))
            mask_v = tf.expand_dims(self.frame_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(layers.mask_logits(att_score, mask = mask_v), axis = 1),(0,2,1))
            self.v2q = tf.matmul(S_, ques_embedding)
            self.q2v = tf.matmul(tf.matmul(S_, S_T), frame_embedding)
            attention_outputs = tf.concat([frame_embedding, self.v2q, frame_embedding * self.v2q, frame_embedding * self.q2v], 2)

        with tf.variable_scope("Model_Encoder_Layer"):

            model_next_layer = conv_utils.linear_mapping(attention_outputs, self.hidden_size,
                                                         dropout=self.dropout,
                                                         var_scope_name="linear_mapping_before_model_layer")
            model_next_layer = transformer.normalize(model_next_layer)
            for i in range(2):
                with tf.variable_scope("stack_%s" % i):
                    model_next_layer = conv_utils.conv_encoder_stack(model_next_layer, [self.hidden_size,self.hidden_size], [3,3],
                                                              {'src': self.dropout,
                                                               'hid': self.dropout}, mode=self.is_training)

                    model_next_layer = transformer.multihead_attention(queries=model_next_layer,
                                                                      keys=model_next_layer,
                                                                      num_units=self.hidden_size,
                                                                      num_heads=4,
                                                                      dropout_rate=1 - self.dropout,
                                                                      is_training=self.is_training,
                                                                      causality=False)

                    model_next_layer = transformer.feedforward(model_next_layer,
                                                               num_units=[2 * self.hidden_size, self.hidden_size],is_training=self.is_training)
            model_outputs = model_next_layer

        with tf.variable_scope("Output_Layer"):

            logit_score = layers.linear_layer_3d(model_outputs, 1, scope_name='output_layer')
            logit_score = tf.squeeze(logit_score, 2)
            logit_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits= logit_score, labels=self.gt_predict)
            avg_logit_loss = tf.reduce_mean(tf.reduce_sum(logit_loss,1))

            variables = tf.trainable_variables()
            print(len(variables))
            regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
            self.test_loss = avg_logit_loss
            self.loss = avg_logit_loss + self.regularization_beta * regularization_cost
            self.frame_score = tf.nn.sigmoid(logit_score)


if __name__ == '__main__':

    config_file = '../configs/config_rnn.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)



