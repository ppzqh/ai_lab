'''
DQN for black-white chess
'''
import tensorflow as tf
import numpy as np
import os
N = 8

class my_state:
    def __init__(self, state, next_state=None, reward=None, action=None):
        self.state = state
        self.nextState = next_state
        self.reward = reward
        self.action = action

class DeepQNetwork:
    def __init__(self, 
                n_actions=64,
                reward_decay=0.9,
                learning_rate=0.01,
                batch_size=32,
                upgrade_target_inter=100,
                memory_size=200,
                e_greedy=0.9):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.gamma = reward_decay
        #epsilon可变化
        self.epsilon = e_greedy
        self.batch_size = batch_size
        #pass

        #设定学习次数
        self.learn_count = 0
        self.upgrade_target_inter = upgrade_target_inter
        self.build_network()

        self.memory = []
        self.memory_index = 0
        self.memory_size = memory_size

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluate_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        #初始化sess
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, 'model/parameter.ckpt')

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, 'model/parameter.ckpt')

    def build_network(self):
        #定义自己的网络
        #默认一次输入一组数据
        self.state = tf.placeholder(tf.float32, shape=[N, N], name='currentState')
        self.nextState = tf.placeholder(tf.float32, shape=[N, N], name='nextState')
        self.reward = tf.placeholder(tf.float32, shape=[1], name='reward')
        self.action = tf.placeholder(tf.int32, shape=[1], name='action')

        '''evaluate_net'''
        with tf.variable_scope('evaluate_net'):
            '''重新决定网络结构'''
            w1_e = tf.Variable(tf.truncated_normal(shape=[N*N, 100], mean=1, stddev=0.1))
            b1_e = tf.Variable(tf.constant(0.1, shape=[100]))
            hidden_e = tf.nn.relu(tf.matmul(tf.reshape(self.state, shape=[-1,N*N]), w1_e) + b1_e)

            w2_e = tf.Variable(tf.truncated_normal(shape=[100, self.n_actions], mean=1, stddev=0.1))
            b2_e = tf.Variable(tf.constant(0.1, shape=[self.n_actions]))
            self.q_eval = tf.matmul(hidden_e, w2_e) + b2_e

        '''target_net'''
        with tf.variable_scope('target_net'):
            w1_t = tf.Variable(tf.truncated_normal(shape=[N*N, 100], mean=1, stddev=0.1))
            b1_t = tf.Variable(tf.constant(0.1, shape=[100]))
            hidden_t = tf.nn.relu(tf.matmul(tf.reshape(self.state, shape=[-1,N*N]), w1_t) + b1_t)

            w2_t = tf.Variable(tf.truncated_normal(shape=[100, self.n_actions], mean=1, stddev=0.1))
            b2_t = tf.Variable(tf.constant(0.1, shape=[self.n_actions]))
            self.q_next = tf.matmul(hidden_t, w2_t) + b2_t

        with tf.variable_scope('q_target'):
            q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store_transition(self, state, nextState, reward, action):
        print("memory:",len(self.memory))
        #memory库已满，从头开始覆盖
        if len(self.memory) == self.memory_size:
            self.memory[self.memory_index] = my_state(state, nextState, reward, action)
            self.memory_index = self.memory_index + 1 if self.memory_index + 1 < self.memory_size else 0
        #向memory库中添加
        else:
            self.memory.append(my_state(state, nextState, reward, action))

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
        else:
            #随机选择
            actions_value = []
        return actions_value

    def learn(self):
        #更新target网络
        if self.learn_count % self.upgrade_target_inter:
            self.sess.run(self.target_replace_op)
            print('target_params_replaced')
        
        #随机梯度下降
        for i in range(self.batch_size):
            #从现有的memory中选择一条
            sample_index = np.random.choice(len(self.memory))
            tmp_memory = self.memory[sample_index]
            train_result, loss = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self.state: tmp_memory.state,
                    self.nextState: tmp_memory.nextState,
                    self.reward: [tmp_memory.reward],
                    self.action: [tmp_memory.action]
                }
            )
        self.learn_count += 1
