import time
import numpy as np
import tensorflow as tf
from model import VI_Block
from utils import fmt_row
from data import process_data_bucks
import pickle
from utils import conv2d_flipkernel

# Data
tf.app.flags.DEFINE_integer('databucks',         10,                      'Size of input data bucks')
tf.app.flags.DEFINE_integer('imsize',         20,                      'Size of input image')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',        200,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              30,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           9,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('ch_q',           8,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      64,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 3,                     'Number of state inputs for each sample (real number, technically is k+1)')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_integer('save_step',   50,                      'Save model every n epochs')
tf.app.flags.DEFINE_boolean('log',            True,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          './log/vintf/',          'Directory to store tensorboard summary')
tf.app.flags.DEFINE_string('model_save_path',          './log/models/',          'Directory to store models')
tf.app.flags.DEFINE_string('data_buck_dir',          './data_buckets/',          'Directory to data')

config = tf.app.flags.FLAGS
imsize = config.imsize
np.random.seed(config.seed)

# symbolic input image tensor with 9 channels
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32,   name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32,   name="S2", shape=[None, config.statebatchsize])
y  = tf.placeholder(tf.int32,   name="y",  shape=[None])

# Construct VI Block
logits, nn = VI_Block(X, S1, S2, config)

# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost)
# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

# Load train and val data
Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_data_bucks(config.imsize,
                                                                                    config.data_buck_dir, config.databucks)

# Launch the graph
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.allow_soft_placement = True
gpu_config.log_device_placement = True

with tf.Session(config=gpu_config) as sess:
    if config.log:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(init)
    model_save_path = config.model_save_path
    saver = tf.train.Saver()

    batch_size = config.batchsize
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
    for epoch in range(int(config.epochs)):
        tstart = time.time()
        avg_err, avg_cost = 0.0, 0.0
        num_batches = int(Xtrain.shape[0]/batch_size)
        # Loop over all batches
        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {X: Xtrain[i:j], S1: S1train[i:j], S2: S2train[i:j],
                    y: ytrain[i * config.statebatchsize:j * config.statebatchsize]}
                _, e_, c_ = sess.run([optimizer, err, cost], feed_dict=fd)
                avg_err += e_
                avg_cost += c_
        # Display logs per epoch step
        if epoch % config.display_step == 0:
            elapsed = time.time() - tstart
            print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
        if epoch != 0 and epoch % config.save_step == 0:
            saver = tf.train.Saver()
            saver.save(sess, model_save_path, epoch)
            print('Save model: ', epoch)
            # Evaluate
            correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, y), dtype=tf.float32))
            acc = accuracy.eval({X: Xtest, S1: S1test, S2: S2test, y: ytest}, session=sess)
            print('Accuracy:', {100 * acc},'%')
            top2_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(nn, y, 2), tf.float32))
            top2_acc = top2_accuracy.eval({X: Xtest, S1: S1test, S2: S2test, y: ytest}, session=sess)
            print('Top2 Accuracy:', {100 * top2_acc},'%')
        if config.log:
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
            summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
            summary_writer.add_summary(summary, epoch)


    saver.save(sess, model_save_path, config.epochs)
    print("Finished training!")

    # Test model
    correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
    # Calculate Top1 accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, y), dtype=tf.float32))
    acc = accuracy.eval({X: Xtest, S1: S1test, S2: S2test, y: ytest}, session=sess)
    print('Top1 Accuracy:', {100 * acc}, '%')

    # Calculate Top1 accuracy
    top2_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(nn, y, 2), tf.float32))
    top2_acc = top2_accuracy.eval({X: Xtest, S1: S1test, S2: S2test, y: ytest}, session=sess)
    print('Top2 Accuracy:', {100 * top2_acc}, '%')
