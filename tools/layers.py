import sys
sys.path.append("..")
import pickle as pkl
import tensorflow as tf
import numpy


def tensor_matmul(ts, mat):
    # ts is a 3d batched tensor, while mat is a 2d parameter matrix
    #matmul = lambda x: tf.matmul(x, mat)
    #output = tf.map_fn(matmul, ts)
    ts_s1 = int(ts.get_shape()[1])
    ts_s2 = int(ts.get_shape()[2])
    mat_s1 = int(mat.get_shape()[1])
    reshaped = tf.reshape(ts, [-1, ts_s2])
    matmul = tf.matmul(reshaped, mat)
    output = tf.reshape(matmul, [-1, ts_s1, mat_s1])
    return output

def count_total_variables():
    n_ts = 0
    n_var = 0
    for ts in tf.trainable_variables():
        n_ts += 1
        ts_size = 1
        for dim in ts.get_shape():
            ts_size *= dim.value
        n_var += ts_size
    print ('Total tensors:', n_ts)
    print ('Total variables:', n_var)

def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def linear_layer(input_ts, output_dim, scope_name):
    # linear layer, input_ts should be a 2d tensor
    with tf.variable_scope(scope_name):
        w = tf.get_variable('w', shape=[input_ts.shape[1], output_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.03))
        b = tf.get_variable('b', shape=[output_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.03))
        linear_output = tf.matmul(input_ts, w) + b
        return linear_output

def linear_layer_3d(input, output_dim, scope_name):
    dim_list = input.get_shape().as_list()
    with tf.variable_scope(scope_name):
        input_ts = tf.reshape(input, [-1, dim_list[-1]])
        w = tf.get_variable('w', shape=[dim_list[-1], output_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())  # initializer=tf.random_normal_initializer(stddev=0.03))
        b = tf.get_variable('b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())  # initializer=tf.random_normal_initializer(stddev=0.03))
        linear_output = tf.matmul(input_ts, w) + b
        linear_output = tf.reshape(linear_output, [-1, dim_list[1],output_dim])
        return linear_output

def static_origin_lstm_layer(input_ts, hidden_dim, scope_name, input_len=None):
    with tf.variable_scope(scope_name):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_output, final_state = tf.contrib.rnn.static_rnn(lstm_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
        return lstm_output, final_state

def dynamic_origin_lstm_layer(input_ts, hidden_dim, scope_name, input_len=None):
    with tf.variable_scope(scope_name):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
        return lstm_output, final_state

# def dynamic_bound_lstm_layer(input_ts, hidden_dim, bound_dim, is_training, scope_name, input_len=None):
#     with tf.variable_scope(scope_name):
#         lstm_cell = BoundLSTMCell(hidden_dim, bound_dim, is_training)
#         lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
#         return lstm_output, final_state

# def dynamic_stacked_bound_lstm_layer(input_ts, hidden_dim, bound_dim, is_training, scope_name, input_len=None):
#     with tf.variable_scope(scope_name):
#         lstm_cell = StackedBoundLSTMCell(hidden_dim, bound_dim, is_training)
#         lstm_output, final_state = tf.nn.dynamic_rnn(lstm_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
#         return lstm_output, final_state

def static_origin_bilstm_layer(input_ts, hidden_dim, scope_name, input_len=None):
    with tf.variable_scope(scope_name):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        bilstm_output, fwlstm_state, bwlstm_state = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
        return bilstm_output, fwlstm_state, bwlstm_state

def dynamic_origin_bilstm_layer(input_ts, hidden_dim, scope_name, input_len=None):
    with tf.variable_scope(scope_name):
        #lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        #lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        #lstm_fw_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_dim)
        #lstm_bw_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_dim)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        bilstm_output, bilstm_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
        return tf.concat(bilstm_output, 2), bilstm_state

# def dynamic_bound_bilstm_layer(input_ts, hidden_dim, bound_dim, is_training, scope_name, input_len=None):
#     with tf.variable_scope(scope_name):
#         lstm_fw_cell = BoundLSTMCell(hidden_dim, bound_dim, is_training)
#         lstm_bw_cell = BoundLSTMCell(hidden_dim, bound_dim, is_training)
#         bilstm_output, bilstm_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_ts, sequence_length=input_len, dtype=tf.float32, scope=scope_name)
#         return tf.concat(bilstm_output, 2), bilstm_state

def scalar_attention_layer(input_ts_list, context_ts, scope_name):
    output_dim = len(input_ts_list)
    input_dim = input_ts_list[0].shape[1]
    context_dim = context_ts.shape[1]
    with tf.variable_scope(scope_name):
        w_c = tf.get_variable('w_c', shape=[context_dim, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        w_i = tf.get_variable('w_i', shape=[input_dim, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        attention_input_list = list()
        for i in range(output_dim):
            attention_input_list.append(tf.matmul(context_ts, w_c) + tf.matmul(input_ts_list[i], w_i))
        # [n_steps * (batch_size, 1)] -> (batch_size, n_steps)
        attention_input = tf.concat(attention_input_list, axis=1)
        w_a = tf.get_variable('w_a', shape=[output_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        attention_state = tf.multiply(w_a, tf.tanh(attention_input))
        attention_score = tf.nn.softmax(attention_state)
        # (batch_size, n_steps) -> (n_steps, batch_size, 2*bilstm_dim)
        attention_score = tf.transpose(attention_score, perm=[1, 0])
        attention_score_broadcast = tf.tile(tf.expand_dims(attention_score, -1), tf.stack([1, 1, input_dim]))
        # (n_steps, batch_size, 2*bilstm_dim), [n_steps * (batch_size, 2*bilstm_dim)] -> [batch_size, 2*bilstm_dim]
        attention_output = tf.reduce_sum(tf.multiply(attention_score_broadcast, tf.stack(input_ts_list)), axis=0)
        return attention_output

def matrix_attention_layer(input_ts, context_ts, att_dim, scope_name, weights_only=False):
    output_dim = int(input_ts.shape[1]) # time_step, L
    input_dim = int(input_ts.shape[2]) # video_dims, k
    context_dim = int(context_ts.shape[1]) # question_dims, c
    #print 'input_ts:', input_ts.shape
    #print 'context_ts:', context_ts.shape
    with tf.variable_scope(scope_name):
        tiled_context = tf.tile(tf.expand_dims(context_ts, 1), tf.stack([1, output_dim, 1]))
        #print 'tiled_context:', tiled_context.shape
        w_c = tf.get_variable('w_c', shape=[context_dim, att_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.003))
        w_i = tf.get_variable('w_i', shape=[input_dim, att_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.001))
        b_i = tf.get_variable('b_i', shape=[att_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        # (batch_size, time_step, att_dim)
        attention_input = tf.tanh(tensor_matmul(input_ts, w_i) + tensor_matmul(tiled_context, w_c) + b_i)
        #print 'attention_input:', attention_input.shape
        w_a = tf.get_variable('w_a', shape=[att_dim, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        # (batch_size, time_step)
        attention_score = tf.nn.softmax(tf.squeeze(tensor_matmul(attention_input, w_a), axis=[2]))
        if weights_only:
            return attention_score
        #attention_score = tf.nn.softmax(tools.tensor_matmul(attention_input, w_a))
        #print 'attention_score:', attention_score.shape
        # (batch_size, output_dim)
        else:
            attention_output = tf.reduce_sum(tf.multiply(input_ts, tf.expand_dims(attention_score, 2)), 1)
            return attention_output, attention_score
        #attention_output = tf.squeeze(tf.matmul(tf.transpose(attention_score, perm=[0,2,1]), input_ts), axis=[1])
        #print 'attention_output:', attention_output.shape

def mask_matrix_attention_layer(input_ts, context_ts, att_dim, mask, scope_name, weights_only=False):
    output_dim = int(input_ts.shape[1]) # time_step, L
    input_dim = int(input_ts.shape[2]) # video_dims, k
    context_dim = int(context_ts.shape[1]) # question_dims, c

    with tf.variable_scope(scope_name):
        tiled_context = tf.tile(tf.expand_dims(context_ts, 1), tf.stack([1, output_dim, 1]))

        w_c = tf.get_variable('w_c', shape=[context_dim, att_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.003))
        w_i = tf.get_variable('w_i', shape=[input_dim, att_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.001))
        b_i = tf.get_variable('b_i', shape=[att_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))

        attention_input = tf.tanh(tensor_matmul(input_ts, w_i) + tensor_matmul(tiled_context, w_c) + b_i)
        w_a = tf.get_variable('w_a', shape=[att_dim, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        attention_score = tf.nn.softmax(tf.multiply(tf.squeeze(tensor_matmul(attention_input, w_a), axis=[2]), mask))
        if weights_only:
            return attention_score
        else:
            attention_output = tf.reduce_sum(tf.multiply(input_ts, tf.expand_dims(attention_score, 2)), 1)
            return attention_output, attention_score

def bilinear_attention_layer(input_ts, context_ts, att_dim, scope_name):
    output_dim = int(input_ts.shape[1]) # time_step, L
    input_dim = int(input_ts.shape[2]) # video_dims, k
    context_dim = int(context_ts.shape[1]) # question_dims, c
    with tf.variable_scope(scope_name):
        reshaped_context = tf.reshape(context_ts, [-1, context_dim, 1])
        p = tf.get_variable('p', shape=[input_dim, context_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.003))
        # (batch_size, time_step, video_dims) * (video_dims, context_dim) * (batch_size, context_dim, 1) -> (batch_size, time_step, 1)
        attention_input = tf.matmul(tensor_matmul(input_ts, p), reshaped_context)
        #print 'attention_input:', attention_input.shape
        # (batch_size, time_step)
        attention_score = tf.nn.softmax(tf.squeeze(attention_input, axis=[2]))
        #print 'attention_score:', attention_score.shape
        # (batch_size, output_dim)
        attention_output = tf.reduce_sum(tf.multiply(input_ts, tf.expand_dims(attention_score, 2)), 1)
        #print 'attention_output:', attention_output.shape
        return attention_output

def collective_matrix_attention_layer(input_ts, context_ts, attention_dim, scope_name, context_len=None, use_maxpooling=True, weights_only=False):
    # input_ts: (batch_size, n_steps, input_dim)
    # context_ts: (batch_size, max_q_n_words, embedding_dim)
    # context_len: (batch_size)
    max_context_len = int(context_ts.shape[1])
    input_dim = int(input_ts.shape[2])
    unstack_context = unstack_layer(context_ts, 'context_unstack', axis=1)
    attention_output_list = list()
    attention_score_list = list()
    for i in range(max_context_len):
        attention_output, attention_score = matrix_attention_layer(input_ts, unstack_context[i], attention_dim, scope_name+'_'+str(i), weights_only=weights_only)
        attention_output_list.append(attention_output)
        attention_score_list.append(attention_score)
    # restack_output: (batch_size, max_q_n_words, input_dim)
    restack_output = restack_layer(attention_output_list, 'context_restack', axis=1)
    attention_score_list = restack_layer(attention_score_list,'score_restack', axis = 1)

    # context_mask: (batch_size, max_q_n_words)
    if context_len is not None:
        context_mask = tf.sequence_mask(context_len, maxlen=max_context_len, dtype=tf.float32)
        context_mask = tf.tile(tf.expand_dims(context_mask, 2), tf.stack([1, 1, input_dim]))
        # masked_output: (batch_size, max_q_n_words, input_dim)
        masked_output = tf.multiply(restack_output, context_mask)
    else:
        masked_output = restack_output

    # max_pooled_output: (batch_size, 1, 1, input_dim)
    if (not weights_only) and use_maxpooling:
        max_pooled_output = tf.nn.max_pool(tf.expand_dims(masked_output,1), ksize=[1, 1, max_context_len, 1], strides=[1, 1, 1, 1], padding='VALID')
        collective_attention_output = tf.squeeze(max_pooled_output, [1,2])
        return collective_attention_output, attention_score_list
    else:
        return masked_output, attention_score_list

def restack_layer(input_ts, scope_name, axis=1):
    with tf.variable_scope(scope_name):
        input_ts = tf.stack(input_ts)
        perm_list = list(range(len(input_ts.shape)))
        perm_list[0], perm_list[axis] = perm_list[axis], perm_list[0]
        restack_output = tf.transpose(input_ts, perm=perm_list)
        return restack_output

def unstack_layer(input_ts, scope_name, axis=1):
    with tf.variable_scope(scope_name):
        perm_list = list(range(len(input_ts.shape)))
        perm_list[0], perm_list[axis] = perm_list[axis], perm_list[0]
        unstack_output = tf.unstack(tf.transpose(input_ts, perm=perm_list))
        return unstack_output

def embedding_layer(input_ts, output_dim, scope_name):
    with tf.variable_scope(scope_name):
        w_e = tf.get_variable('w_e', shape=[input_ts.shape[1], output_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # initializer=tf.random_normal_initializer(stddev=0.01))
        embedding_output = tf.nn.embedding_lookup(w_e, input_ts)
        #embedding_output = tf.reduce_mean(embedding_output, reduction_indices=1)
        return embedding_output

def w2v_embedding_layer(input_ts, output_dim, scope_name):
    with tf.variable_scope(scope_name):
        embeddings = pkl.load(open('/home/fenixlin/data/video_qa/yqf/word2vec.pkl'))
        embeddings = tf.constant(embeddings, dtype=tf.float32)
        embedding_output = tf.nn.embedding_lookup(embeddings, input_ts)
        #embedding_output = tf.reduce_mean(embedding_output, reduction_indices=1)
        return embedding_output
