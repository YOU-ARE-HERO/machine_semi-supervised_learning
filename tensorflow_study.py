# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import pickle
import gym


# 构造字典
# ff = h5py.File(r'C:\Users\jessefeng\Desktop\writer.hdf5','w')
# dic = {'a':3,'h':45,'c':78}
# for key in dic:
#     print (key)
#     print (dic[key])
#     ff.create_dataset(key,data=dic[key])
# ff.close()
#
# h = h5py.File(r'C:\Users\jessefeng\Desktop\writer.hdf5')
# flower_captions = {}
# for ds in h.items():
#     print (ds[0],np.array(ds[1]))
#     # flower_captions[ds[0]] = np.array(ds[1])

# dic = {'a':3,'h':45,'c':78,'k':1000}
# for i,j in enumerate(dic):
#     print (i,dic[j])
#
# dic_items = list(dic.items()) #转化成为列表
# print (type(dic_items))
# dic_items_trans = dic_items[0:2]
# print (dic_items_trans)
#
# diccc = dict(dic_items_trans)
# print (diccc)
# env = gym.make('CartPole-v0')
# env.reset()
#
# ## 网络参数 ##
# H =50 # 隐含层节点数
# batch_size = 25
# learning_rate = 1e-1
# D = 4 # 环境信息observation的维度
# gamma = 0.99
#
#
# ## 网络结构 ##
# observations = tf.placeholder(tf.float32,[None,D],name = "input_x")
# w1 = tf.get_variable("w1",shape=[D,H],initializer=tf.contrib.layers.xavier_initializer())
# layer1 = tf.nn.relu(tf.matmul(observations,w1))
#
# w2 = tf.get_variable("w2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
# score = tf.matmul(layer1,w2)
# probability = tf.nn.sigmoid(score)
#
# ## 优化参数 ##
# adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
# w1grad = tf.placeholder(tf.float32,name="batch_grad1")
# w2grad = tf.placeholder(tf.float32,name="batch_grad2")
# batch_grad = [w1grad,w2grad]
#
# tvars = tf.trainable_variables() # 获取网络中全部可训练的参数tvars
# updategrads = adam.apply_gradients(zip(batch_grad,tvars))
#
#
# ## 未来奖励 ##
# def discount_rewards(r):
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(range(r.size)):
#         running_add = running_add*gamma + r[t] # 直接获取的reward加上gamma乘以之后获得报酬
#         discounted_r[t] = running_add
#     return discounted_r
#
# ## 定义损失函数 ##
# input_y = tf.placeholder(tf.float32,[None,1],name="input_y")
# advantages = tf.placeholder(tf.float32,name="reward_singal")
# loglik = tf.log(input_y*(input_y - probability) + (1-input_y)*(input_y + probability))
# loss = -tf.reduce_mean(loglik * advantages)
#
# tvars = tf.trainable_variables()
# newGrads = tf.gradients(loss,tvars)
#
# xs,ys,drs = [],[],[]
# reward_sum = 0
# episode_number = 1
# total_episodes = 10000
#
# with tf.Session() as sess:
#     rendering = False
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     observation = env.reset()
#
#     gradBuffer = sess.run(tvars)
#     for ix,grad in enumerate(gradBuffer):
#         gradBuffer[ix] = grad * 0
#
#     while episode_number <= total_episodes:
#         if reward_sum/batch_size >100 or rendering ==True:
#             env.render()
#             rendering = True
#
#         x = np.reshape(observation,[1,D])
#
#         tfprob = sess.run(probability, feed_dict={observations: x})
#         action = 1 if np.random.uniform()< tfprob else 0
#
#         xs.append(x)
#         y = 1-action
#         ys.append(y)
#
#         observation,reward,done,info = env.step(action)
#         reward_sum +=reward
#         drs.append(reward)
#
#         if done:
#             episode_number += 1
#             epx = np.vstack(xs)
#             epy = np.vstack(ys)
#             epr = np.vstack(drs)
#
#             xs,yx,drs = [],[],[]
#             discounted_epr = discount_rewards(epr)
#             discounted_epr -=np.mean(discounted_epr)
#             discounted_epr /= np.std(discounted_epr)
#
#             tGrad = sess.run(newGrads,feed_dict={observations:epx,input_y:epy,advantages:discounted_epr})
#             for ix,grad in enumerate(tGrad):
#                 gradBuffer[ix] += grad
#
#             if episode_number % batch_size == 0:
#                 sess.run(updategrads,feed_dict={w1grad:gradBuffer[0],w2grad:gradBuffer[1]})
#
#                 for ix,grad in enumerate(gradBuffer):
#                     gradBuffer[ix] = grad * 0
#                 print ("Average reward for episode %d : %f." %(episode_number,reward_sum/batch_size))
#
#                 if reward_sum/batch_size > 200:
#                     print ("Task solved in",episode_number,"episodes!")
#                     break
#                 reward_sum = 0
#             observation = env.reset()
#
# from sklearn.tree import DecisionTreeClassifier


import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sess = tf.InteractiveSession()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot=True)


learning_rate = 0.001
batch_size = 128

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weight, biases):
    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x,n_steps,0)
    outputs = list()
    lstm = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    state = (tf.zeros([n_steps, n_hidden]),) * 2
    sess.run(state) #可以将其中的值打印出来
    with tf.variable_scope("myrnn2") as scope:
        for i in range(n_steps - 1):
            print ("start i:",i)
            if i > 0:
                scope.reuse_variables() # 复用lstm中参数
                print (u"i等于：",i)
            output, state = lstm(x[i], state)
            print ('output:',output)
            print('state:', state)
            outputs.append(output)
        print ('total output:',output)
    final = tf.matmul(outputs[-1], weights['out']) + biases['out']
    print ('final:',final) # 28*10
    return final


pred = RNN(x ,weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
print ('cost:',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess.run(init)
for step in range(20000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    if step % 50 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print( "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
print ("Optimization Finished!")









