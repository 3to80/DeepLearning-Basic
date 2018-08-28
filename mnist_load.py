import tensorflow as tf


saver = tf.train.Saver()



final_model_path = "./my_mnist_model"

with tf.Session() as sess:
    saver.restore(sess, final_model_path)




