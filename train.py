import os
import tensorflow as tf
import input_data
import resnet_50

BATCH_SIZE = 100
IMG_H = 224
IMG_W = 224
CLASS_NUM = 2000
LEARNING_RATE = 0.001
MAX_STEP = 100000
LOG_DIR = "/home/rvlab/program/ResNet/log/"
SAVE_DIR = "/home/rvlab/program/ResNet/log2/"

# #######################
#  write tfrecords file
# #######################
# images, labels = input_data.get_files("/home/rvlab/program/0data/CACD/")
# index = int(len(images) * 4 / 5)
# tra_images = images[:index]
# tra_labels = labels[:index]
# val_images = images[index:]
# val_labels = labels[index:]
# input_data.write_tfrecord(tra_images, tra_labels, "/home/rvlab/program/0data/CACD_tra.tfrecords")
# input_data.write_tfrecord(val_images, val_labels, "/home/rvlab/program/0data/CACD_val.tfrecords")


# ######################
#  read tfrecords file
# ######################
with tf.name_scope("input"):
    tra_img_batch, tra_lab_batch = input_data.read_tfrecord("/home/rvlab/program/0data/CACD_tra.tfrecords",
                                                            batch_size=BATCH_SIZE,
                                                            image_H=IMG_H,
                                                            image_W=IMG_W)
    val_img_batch, val_lab_batch = input_data.read_tfrecord("/home/rvlab/program/0data/CACD_val.tfrecords",
                                                            batch_size=BATCH_SIZE,
                                                            image_H=IMG_H,
                                                            image_W=IMG_W)

global_step = tf.Variable(0, trainable=False)
x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])
logits = resnet_50.inference(x, CLASS_NUM)
loss = resnet_50.losses(logits, y)
acc = resnet_50.evaluation(logits, y)
train_op = resnet_50.training(loss, LEARNING_RATE, global_step)
summary_op = tf.summary.merge_all()
#######################
#  fine-tuning1
#######################
# var_list = []
# for layer_name in []:
#     with tf.variable_scope(layer_name, reuse=True):
#         var_list += [tf.get_variable("weights")]
#         var_list += [tf.get_variable("biases")]
saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#######################
#  fine-tuning2
#######################
# print("Reading checkpoints...")
# ckpt = tf.train.get_checkpoint_state(LOG_DIR)
# if ckpt and ckpt.model_checkpoint_path:
#     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print('Loading success, global_step is %s' % global_step)
# else:
#     print('No checkpoint file found')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
tra_summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR, "tra"), sess.graph)
val_summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR, "val"), sess.graph)

for step in range(MAX_STEP):
    data_batch, label_batch = sess.run([tra_img_batch, tra_lab_batch])
    _, summary_str, tra_loss, tra_acc = sess.run([train_op, summary_op, loss, acc],
                                                 feed_dict={x: data_batch,
                                                            y: label_batch})
    if step % 100 == 0 or (step + 1) == MAX_STEP:
        tra_summary_writer.add_summary(summary_str, step)
        print("Train Step %d, loss: %.4f, accuracy: %.4f" % (step, tra_loss, tra_acc))
    if step % 1000 == 0 or (step + 1) == MAX_STEP:
        data_batch, label_batch = sess.run([val_img_batch, val_lab_batch])
        summary_str, val_loss, val_acc = sess.run([summary_op, loss, acc],
                                                  feed_dict={x: data_batch,
                                                             y: label_batch})
        val_summary_writer.add_summary(summary_str, step)
        print("## Val Step %d, loss: %.4f, accuracy: %.4f" % (step, val_loss, val_acc))
    if (step + 1) % 10000 == 0 or (step + 1) == MAX_STEP:
        saver.save(sess, os.path.join(SAVE_DIR, "model.ckpt"), global_step=step)

coord.request_stop()
coord.join(threads)
sess.close()
