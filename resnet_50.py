import tensorflow as tf


def conv(layer_name, x, in_channel, out_channel, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True):
    with tf.variable_scope(layer_name):
        weights = tf.get_variable(name="weights",
                                  shape=[ksize[0],
                                         ksize[1],
                                         in_channel,
                                         out_channel],
                                  trainable=is_train,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

        biases = tf.get_variable(name="biases",
                                 shape=[out_channel],
                                 trainable=is_train,
                                 initializer=tf.constant_initializer(0.1))

        x = tf.nn.conv2d(x, weights, stride, padding="SAME")
        x = tf.nn.bias_add(x, biases)
    return x


def pool_max(layer_name, x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
    with tf.variable_scope(layer_name):
        x = tf.nn.max_pool(x, ksize=ksize, strides=stride, padding="SAME")
    return x


def pool_ave(layer_name, x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1]):
    with tf.variable_scope(layer_name):
        x = tf.nn.avg_pool(x, ksize=ksize, strides=stride, padding="SAME")
    return x


def fc(layer_name, x, in_channel, out_channel, is_train=True):
    with tf.variable_scope(layer_name):
        x = tf.reshape(x, shape=[-1, in_channel])
        weights = tf.get_variable(name="weights",
                                  shape=[in_channel, out_channel],
                                  trainable=is_train,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))

        biases = tf.get_variable(name="biases",
                                 shape=[out_channel],
                                 trainable=is_train,
                                 initializer=tf.constant_initializer(0.1))

        x = tf.matmul(x, weights)
        x = tf.nn.bias_add(x, biases)
    return x


def relu(layer_name, x):
    with tf.variable_scope(layer_name):
        x = tf.nn.relu(x)
    return x


def batch_norm(layer_name, x):
    with tf.variable_scope(layer_name):
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        x = tf.nn.batch_normalization(x=x,
                                      mean=batch_mean,
                                      variance=batch_var,
                                      offset=None,
                                      scale=None,
                                      variance_epsilon=1e-3)
    return x


def inference(x, class_num):
    '''
    :param x: [batch_size, height, width, channel]
    :param class_num:
    :return:
    '''
    # input data: [batch_size, 224, 224, 3]
    conv1 = conv("conv1", x, 3, 64, ksize=[7, 7], stride=[1, 2, 2, 1], is_train=True)
    conv1 = batch_norm("bn_conv1", conv1)
    conv1 = relu("conv1_relu", conv1)
    pool1 = pool_max("pool1", conv1, ksize=[1, 3, 3, 1], stride=[1, 2, 2, 1])

    res2a_branch1 = conv("res2a_branch1", pool1, 64, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2a_branch1 = batch_norm("bn2a_branch1", res2a_branch1)

    res2a_branch2a = conv("res2a_branch2a", pool1, 64, 64, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2a_branch2a = batch_norm("bn2a_branch2a", res2a_branch2a)
    res2a_branch2a = relu("res2a_branch2a_relu", res2a_branch2a)
    res2a_branch2b = conv("res2a_branch2b", res2a_branch2a, 64, 64, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res2a_branch2b = batch_norm("bn2a_branch2b", res2a_branch2b)
    res2a_branch2b = relu("res2a_branch2b_relu", res2a_branch2b)
    res2a_branch2c = conv("res2a_branch2c", res2a_branch2b, 64, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2a_branch2c = batch_norm("bn2a_branch2c", res2a_branch2c)
    with tf.variable_scope("res2a"):
        res2a = tf.add(res2a_branch1, res2a_branch2c)
    res2a = relu("res2a_relu", res2a)
    res2b_branch2a = conv("res2b_branch2a", res2a, 256, 64, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2b_branch2a = batch_norm("bn2b_branch2a", res2b_branch2a)
    res2b_branch2a = relu("res2b_branch2a_relu", res2b_branch2a)
    res2b_branch2b = conv("res2b_branch2b", res2b_branch2a, 64, 64, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res2b_branch2b = batch_norm("bn2b_branch2b", res2b_branch2b)
    res2b_branch2b = relu("res2b_branch2b_relu", res2b_branch2b)
    res2b_branch2c = conv("res2b_branch2c", res2b_branch2b, 64, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2b_branch2c = batch_norm("bn2b_branch2c", res2b_branch2c)
    with tf.variable_scope("res2b"):
        res2b = tf.add(res2a, res2b_branch2c)
    res2b = relu("res2b_relu", res2b)
    res2c_branch2a = conv("res2c_branch2a", res2b, 256, 64, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2c_branch2a = batch_norm("bn2c_branch2a", res2c_branch2a)
    res2c_branch2a = relu("res2c_branch2a_relu", res2c_branch2a)
    res2c_branch2b = conv("res2c_branch2b", res2c_branch2a, 64, 64, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res2c_branch2b = batch_norm("bn2c_branch2b", res2c_branch2b)
    res2c_branch2b = relu("res2c_branch2b_relu", res2c_branch2b)
    res2c_branch2c = conv("res2c_branch2c", res2c_branch2b, 64, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res2c_branch2c = batch_norm("bn2c_branch2c", res2c_branch2c)
    with tf.variable_scope("res2c"):
        res2c = tf.add(res2b, res2c_branch2c)
    res2c = relu("res2c_relu", res2c)

    res3a_branch1 = conv("res3a_branch1", res2c, 256, 512, ksize=[1, 1], stride=[1, 2, 2, 1], is_train=True)
    res3a_branch1 = batch_norm("bn3a_branch1", res3a_branch1)

    res3a_branch2a = conv("res3a_branch2a", res2c, 256, 128, ksize=[1, 1], stride=[1, 2, 2, 1], is_train=True)
    res3a_branch2a = batch_norm("bn3a_branch2a", res3a_branch2a)
    res3a_branch2a = relu("res3a_branch2a_relu", res3a_branch2a)
    res3a_branch2b = conv("res3a_branch2b", res3a_branch2a, 128, 128, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res3a_branch2b = batch_norm("bn3a_branch2b", res3a_branch2b)
    res3a_branch2b = relu("res3a_branch2b_relu", res3a_branch2b)
    res3a_branch2c = conv("res3a_branch2c", res3a_branch2b, 128, 512, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3a_branch2c = batch_norm("bn3a_branch2c", res3a_branch2c)
    with tf.variable_scope("res3a"):
        res3a = tf.add(res3a_branch1, res3a_branch2c)
    res3a = relu("res3a_relu", res3a)
    res3b_branch2a = conv("res3b_branch2a", res3a, 512, 128, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3b_branch2a = batch_norm("bn3b_branch2a", res3b_branch2a)
    res3b_branch2a = relu("res3b_branch2a_relu", res3b_branch2a)
    res3b_branch2b = conv("res3b_branch2b", res3b_branch2a, 128, 128, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res3b_branch2b = batch_norm("bn3b_branch2b", res3b_branch2b)
    res3b_branch2b = relu("res3b_branch2b_relu", res3b_branch2b)
    res3b_branch2c = conv("res3b_branch2c", res3b_branch2b, 128, 512, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3b_branch2c = batch_norm("bn3b_branch2c", res3b_branch2c)
    with tf.variable_scope("res3b"):
        res3b = tf.add(res3a, res3b_branch2c)
    res3b = relu("res3b_relu", res3b)
    res3c_branch2a = conv("res3c_branch2a", res3b, 512, 128, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3c_branch2a = batch_norm("bn3c_branch2a", res3c_branch2a)
    res3c_branch2a = relu("res3c_branch2a_relu", res3c_branch2a)
    res3c_branch2b = conv("res3c_branch2b", res3c_branch2a, 128, 128, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res3c_branch2b = batch_norm("bn3c_branch2b", res3c_branch2b)
    res3c_branch2b = relu("res3c_branch2b_relu", res3c_branch2b)
    res3c_branch2c = conv("res3c_branch2c", res3c_branch2b, 128, 512, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3c_branch2c = batch_norm("bn3c_branch2c", res3c_branch2c)
    with tf.variable_scope("res3c"):
        res3c = tf.add(res3b, res3c_branch2c)
    res3c = relu("res3c_relu", res3c)
    res3d_branch2a = conv("res3d_branch2a", res3c, 512, 128, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3d_branch2a = batch_norm("bn3d_branch2a", res3d_branch2a)
    res3d_branch2a = relu("res3d_branch2a_relu", res3d_branch2a)
    res3d_branch2b = conv("res3d_branch2b", res3d_branch2a, 128, 128, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res3d_branch2b = batch_norm("bn3d_branch2b", res3d_branch2b)
    res3d_branch2b = relu("res3d_branch2b_relu", res3d_branch2b)
    res3d_branch2c = conv("res3d_branch2c", res3d_branch2b, 128, 512, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res3d_branch2c = batch_norm("bn3d_branch2c", res3d_branch2c)
    with tf.variable_scope("res3d"):
        res3d = tf.add(res3c, res3d_branch2c)
    res3d = relu("res3d_relu", res3d)

    res4a_branch1 = conv("res4a_branch1", res3d, 512, 1024, ksize=[1, 1], stride=[1, 2, 2, 1], is_train=True)
    res4a_branch1 = batch_norm("bn4a_branch1", res4a_branch1)

    res4a_branch2a = conv("res4a_branch2a", res3d, 512, 256, ksize=[1, 1], stride=[1, 2, 2, 1], is_train=True)
    res4a_branch2a = batch_norm("bn4a_branch2a", res4a_branch2a)
    res4a_branch2a = relu("res4a_branch2a_relu", res4a_branch2a)
    res4a_branch2b = conv("res4a_branch2b", res4a_branch2a, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res4a_branch2b = batch_norm("bn4a_branch2b", res4a_branch2b)
    res4a_branch2b = relu("res4a_branch2b_relu", res4a_branch2b)
    res4a_branch2c = conv("res4a_branch2c", res4a_branch2b, 256, 1024, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4a_branch2c = batch_norm("bn4a_branch2c", res4a_branch2c)
    with tf.variable_scope("res4a"):
        res4a = tf.add(res4a_branch1, res4a_branch2c)
    res4a = relu("res4a_relu", res4a)
    res4b_branch2a = conv("res4b_branch2a", res4a, 1024, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4b_branch2a = batch_norm("bn4b_branch2a", res4b_branch2a)
    res4b_branch2a = relu("res4b_branch2a_relu", res4b_branch2a)
    res4b_branch2b = conv("res4b_branch2b", res4b_branch2a, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res4b_branch2b = batch_norm("bn4b_branch2b", res4b_branch2b)
    res4b_branch2b = relu("res4b_branch2b_relu", res4b_branch2b)
    res4b_branch2c = conv("res4b_branch2c", res4b_branch2b, 256, 1024, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4b_branch2c = batch_norm("bn4b_branch2c", res4b_branch2c)
    with tf.variable_scope("res4b"):
        res4b = tf.add(res4a, res4b_branch2c)
    res4b = relu("res4b_relu", res4b)
    res4c_branch2a = conv("res4c_branch2a", res4b, 1024, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4c_branch2a = batch_norm("bn4c_branch2a", res4c_branch2a)
    res4c_branch2a = relu("res4c_branch2a_relu", res4c_branch2a)
    res4c_branch2b = conv("res4c_branch2b", res4c_branch2a, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res4c_branch2b = batch_norm("bn4c_branch2b", res4c_branch2b)
    res4c_branch2b = relu("res4c_branch2b_relu", res4c_branch2b)
    res4c_branch2c = conv("res4c_branch2c", res4c_branch2b, 256, 1024, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4c_branch2c = batch_norm("bn4c_branch2c", res4c_branch2c)
    with tf.variable_scope("res4c"):
        res4c = tf.add(res4b, res4c_branch2c)
    res4c = relu("res4c_relu", res4c)
    res4d_branch2a = conv("res4d_branch2a", res4c, 1024, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4d_branch2a = batch_norm("bn4d_branch2a", res4d_branch2a)
    res4d_branch2a = relu("res4d_branch2a_relu", res4d_branch2a)
    res4d_branch2b = conv("res4d_branch2b", res4d_branch2a, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res4d_branch2b = batch_norm("bn4d_branch2b", res4d_branch2b)
    res4d_branch2b = relu("res4d_branch2b_relu", res4d_branch2b)
    res4d_branch2c = conv("res4d_branch2c", res4d_branch2b, 256, 1024, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4d_branch2c = batch_norm("bn4d_branch2c", res4d_branch2c)
    with tf.variable_scope("res4d"):
        res4d = tf.add(res4c, res4d_branch2c)
    res4d = relu("res4d_relu", res4d)
    res4e_branch2a = conv("res4e_branch2a", res4d, 1024, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4e_branch2a = batch_norm("bn4e_branch2a", res4e_branch2a)
    res4e_branch2a = relu("res4e_branch2a_relu", res4e_branch2a)
    res4e_branch2b = conv("res4e_branch2b", res4e_branch2a, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res4e_branch2b = batch_norm("bn4e_branch2b", res4e_branch2b)
    res4e_branch2b = relu("res4e_branch2b_relu", res4e_branch2b)
    res4e_branch2c = conv("res4e_branch2c", res4e_branch2b, 256, 1024, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4e_branch2c = batch_norm("bn4e_branch2c", res4e_branch2c)
    with tf.variable_scope("res4e"):
        res4e = tf.add(res4d, res4e_branch2c)
    res4e = relu("res4e_relu", res4e)
    res4f_branch2a = conv("res4f_branch2a", res4e, 1024, 256, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4f_branch2a = batch_norm("bn4f_branch2a", res4f_branch2a)
    res4f_branch2a = relu("res4f_branch2a_relu", res4f_branch2a)
    res4f_branch2b = conv("res4f_branch2b", res4f_branch2a, 256, 256, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res4f_branch2b = batch_norm("bn4f_branch2b", res4f_branch2b)
    res4f_branch2b = relu("res4f_branch2b_relu", res4f_branch2b)
    res4f_branch2c = conv("res4f_branch2c", res4f_branch2b, 256, 1024, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res4f_branch2c = batch_norm("bn4f_branch2c", res4f_branch2c)
    with tf.variable_scope("res4f"):
        res4f = tf.add(res4e, res4f_branch2c)
    res4f = relu("res4f_relu", res4f)

    res5a_branch1 = conv("res5a_branch1", res4f, 1024, 2048, ksize=[1, 1], stride=[1, 2, 2, 1], is_train=True)
    res5a_branch1 = batch_norm("bn5a_branch1", res5a_branch1)

    res5a_branch2a = conv("res5a_branch2a", res4f, 1024, 512, ksize=[1, 1], stride=[1, 2, 2, 1], is_train=True)
    res5a_branch2a = batch_norm("bn5a_branch2a", res5a_branch2a)
    res5a_branch2a = relu("res5a_branch2a_relu", res5a_branch2a)
    res5a_branch2b = conv("res5a_branch2b", res5a_branch2a, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res5a_branch2b = batch_norm("bn5a_branch2b", res5a_branch2b)
    res5a_branch2b = relu("res5a_branch2b_relu", res5a_branch2b)
    res5a_branch2c = conv("res5a_branch2c", res5a_branch2b, 512, 2048, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res5a_branch2c = batch_norm("bn5a_branch2c", res5a_branch2c)
    with tf.variable_scope("res5a"):
        res5a = tf.add(res5a_branch1, res5a_branch2c)
    res5a = relu("res5a_relu", res5a)
    res5b_branch2a = conv("res5b_branch2a", res5a, 2048, 512, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res5b_branch2a = batch_norm("bn5b_branch2a", res5b_branch2a)
    res5b_branch2a = relu("res5b_branch2a_relu", res5b_branch2a)
    res5b_branch2b = conv("res5b_branch2b", res5b_branch2a, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res5b_branch2b = batch_norm("bn5b_branch2b", res5b_branch2b)
    res5b_branch2b = relu("res5b_branch2b_relu", res5b_branch2b)
    res5b_branch2c = conv("res5b_branch2c", res5b_branch2b, 512, 2048, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res5b_branch2c = batch_norm("bn5b_branch2c", res5b_branch2c)
    with tf.variable_scope("res5b"):
        res5b = tf.add(res5a, res5b_branch2c)
    res5b = relu("res5b_relu", res5b)
    res5c_branch2a = conv("res5c_branch2a", res5b, 2048, 512, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res5c_branch2a = batch_norm("bn5c_branch2a", res5c_branch2a)
    res5c_branch2a = relu("res5c_branch2a_relu", res5c_branch2a)
    res5c_branch2b = conv("res5c_branch2b", res5c_branch2a, 512, 512, ksize=[3, 3], stride=[1, 1, 1, 1], is_train=True)
    res5c_branch2b = batch_norm("bn5c_branch2b", res5c_branch2b)
    res5c_branch2b = relu("res5c_branch2b_relu", res5c_branch2b)
    res5c_branch2c = conv("res5c_branch2c", res5c_branch2b, 512, 2048, ksize=[1, 1], stride=[1, 1, 1, 1], is_train=True)
    res5c_branch2c = batch_norm("bn5c_branch2c", res5c_branch2c)
    with tf.variable_scope("res5c"):
        res5c = tf.add(res5b, res5c_branch2c)
    res5c = relu("res5c_relu", res5c)

    pool5 = pool_ave("pool5", res5c, ksize=[1, 7, 7, 1], stride=[1, 7, 7, 1])
    x = fc("fc6", pool5, 1*1*2048, class_num, is_train=True)

    return x


def losses(logits, labels):
    '''
    :param logits: [batch_size, class_num]
    :param labels: [batch_size]
    :return:
    '''
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("loss", loss)
    return loss


def evaluation(logits, labels):
    '''
    :param logits: [batch_size, class_num]
    :param labels: [batch_size]
    :return:
    '''
    with tf.name_scope("accuracy"):
        predictions = tf.nn.softmax(logits)
        correct = tf.nn.in_top_k(predictions, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar("accuracy", accuracy)
    return accuracy


def training(loss, learning_rate, global_step):
    '''
    :param loss:
    :param learning_rate:
    :return:
    '''
    with tf.name_scope("optimizer"):
        decayed_lr = tf.train.exponential_decay(learning_rate=learning_rate,
                                                global_step=global_step,
                                                decay_steps=10000,
                                                decay_rate=0.5,
                                                staircase=True)

        optimizer = tf.train.AdamOptimizer(decayed_lr)
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for gradients trainable variables and gradients
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradients", grad)

    return train_op
