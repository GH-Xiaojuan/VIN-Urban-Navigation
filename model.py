import numpy as np
import tensorflow as tf
from utils import conv2d_flipkernel


def VI_Block(X, S1, S2, config):
    k = config.k  # Number of value iterations performed
    ch_i = config.ch_i  # Channels in input layer
    ch_q = config.ch_q  # Channels in q layer (~actions)
    state_batch_size = config.statebatchsize

    w_r = tf.Variable(np.random.randn(1, 1, ch_i, ch_q) * 0.01, dtype=tf.float32)
    w_v = tf.Variable(np.random.randn(3, 3, 1, ch_q) * 0.01, dtype=tf.float32)
    w_o = tf.Variable(np.random.randn(ch_q, ch_q) * 0.01, dtype=tf.float32)

    # initial conv layer
    q = conv2d_flipkernel(X, w_r, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k - 1):
        q1 = conv2d_flipkernel(X, w_r, name="q1")
        q2 = conv2d_flipkernel(v, w_v, name="q2")
        q = q1 + q2
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q1 = conv2d_flipkernel(X, w_r, name="q1")
    q2 = conv2d_flipkernel(v, w_v, name="q2")
    q = q1 + q2

    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    #       transpose calls with better indexing for gather_nd

    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")

    return logits, output