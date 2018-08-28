import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import os

tf.reset_default_graph()

mnist = input_data.read_data_sets("../mnist_data/data/")


# data 불러오기
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

# layer 정보
n_inputs = X_train.shape[1]  # MNIST
n_hidden1 = 196
n_hidden2 = 196
n_outputs = 10





# 시각화 하려는 데이터 : iteration에 따른 loss 값, accuracy e
# X, y는 training sample
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int64, shape = (None), name="y")

## BN을 위한

training = tf.placeholder_with_default(
    False, shape=(), name='training')

from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization,
                              training=training, momentum=0.9)




with tf.name_scope("dnn"):
    # X에 X_batch, mean 은 X_batch의 평균, var 도 ~ , offset 은 normal 값에서 초기화? , scale 도 normal 값에ㅓ 초기화?


    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
    bn1_act = tf.nn.elu(bn1)

    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
    bn2_act = tf.nn.elu(bn2)

    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
    logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                           momentum=0.9)


# tensorboard init
tf.summary.merge_all()

# tensor board에 loss를 적을거야
with tf.name_scope("loss"):
    # softmax -> multi- bernoulli  dist
    xentropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels= y)
    loss= tf.reduce_mean(xentropy, name= "loss")
    loss_str = tf.summary.scalar(name="log_loss", tensor=loss)



with tf.name_scope("train"):
    threshold = 1.0
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    grads_and_vars = optimizer.compute_gradients(loss)

    capped_gvs = [
        (tf.clip_by_value(grad, -threshold, threshold), var)
        for grad, var in grads_and_vars]


    training_op = optimizer.apply_gradients(capped_gvs)



with tf.name_scope("eval"):
    # logits, y의 i 번째 요소에 대해
    # ith_ logits (M , classs_num) 에서 가장 높은 값이 y_i 랑 같으면 true,
    correct = tf.nn.in_top_k(logits, y, 1) # (M, 1) vector가 생성
    accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)

# 사전 준비

# 1) initializer : tensor 들 초기화
# 2) saver : 체크포인트 저장하기 위해
#   - check_point를 저장할 경로
# 3) tensorboard를 위한 FileWriter
#   - tensorboard가 사용할 log를 저장할 파일 경로가 필요
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "mnist_log"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

file_writer = tf.summary.FileWriter(log_dir("dnn_clf"), graph=tf.get_default_graph())

# correct 띄울 거는 valid data로 할거니까
X_valid = mnist.validation.images
y_valid = mnist.validation.labels

m, n = X_train.shape


###### 세션 실행

# session traing param
n_epochs = 200
batch_size = 40
n_batch = m//batch_size

current_path = os.getcwd()
print(current_path)


checkpoint_path = current_path + "/tmp/my_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = current_path + "/model/my_mnist_model"


# 조기 종료를 위해
best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

# 정규화
means = mnist.train.images.mean(axis=0, keepdims=True)
stds = mnist.train.images.std(axis=0, keepdims=True) + 1e-10



extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    start_epoch = 0

    if os.path.isfile(checkpoint_epoch_path):
        # epoch 값을 가져와야 된다.
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print(" 이전 훈련이 중지되었습니다. 에포크 {}에서 시작합니다".format(start_epoch))
        saver.restore(sess, checkpoint_path)
    else:
        sess.run(init)


    for epoch in range(start_epoch, n_epochs):
        # epoch 마다 loss , accuracy를 보여 줄 것

        for batch_idx in range(n_batch):
            X_batch, y_batch = mnist.train.next_batch(batch_size=batch_size)
            # get X_batch 의 각 feature 들의 평균과 variance를 구해야됨

            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})

        accuracy_val, loss_val, accuracy_summary_val, loss_summary_val=\
            sess.run([accuracy, loss, accuracy_summary, loss_str],
                     feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(summary=accuracy_summary_val, global_step=epoch)
        file_writer.add_summary(summary=loss_summary_val, global_step=epoch)

        # 5번마다 진행되는 과정을 보여준다. loss, accuracy 값
        # 중간 저장을 해준다. 중간 저장은 현재까지 수행한 epoch 정보 epoch 저장 파일에
        if epoch% 5 ==0:
            print("에포크: " , epoch, " loss val : ", loss_val,
                  "accuracy_val: ", accuracy_val)
            checkpoint_path= saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch+1))
            if loss_val < best_loss:
                final_model_path = saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                # 전체 epoch가 50 이야. 그리고 50번을 수행했는데 더이상 좋아지지 않으면 바로 끝내는거
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("조기 종료")
                    break
# 완벽하게 끝나면 epcoh_path는 제거
os.remove(checkpoint_epoch_path)


print("####################### 훈련 성공 #####################")


# 훈련한걸 가지고 와서 써보기
# 이때는 test set으로 해야됨!


with tf.Session() as sess:
    # saver.restore(sess, final_model_path)
    # final_accuracy_val= \
    #     sess.run(accuracy_val, feed_dict={X: X_test, y: y_test})
    #
    saver.restore(sess, final_model_path)
    final_accuracy_val, final_loss_val = sess.run([accuracy, loss], feed_dict={X: X_test, y: y_test})



print("최종 acc : ", final_accuracy_val, " 최종 loss : ", final_loss_val)




