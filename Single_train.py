from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Multi_env
import os
from pool import ReplayMemory
import sys

my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################
pos_1 = [i for i in [3.5, 3.5+ 3.5, 250 + 3.5, 250 + 3.5 + 3.5, 500 + 3.5, 500 + 3.5 + 3.5]]
pos_2 = [i for i in [250 - 3.5 - 3.5, 250 - 3.5, 500 - 3.5 - 3.5, 500 - 3.5, 750 - 3.5 - 3.5, 750 - 3.5]]
pos_3 = [i for i in [3.5, 3.5+ 3.5, 433 + 3.5, 433 + 3.5 + 3.5, 866 + 3.5, 866 + 3.5 + 3.5]]
pos_4 = [i for i in [433 - 3.5 - 3.5, 433 - 3.5, 866 - 3.5 - 3.5, 866 - 3.5, 1299 - 3.5 - 3.5, 1299 - 3.5]]

width = 750
height = 1298

is_train = 1
is_test = 1 - is_train

label = 'single'

n_user = 4
n_neighbor = 1
n_RB = n_user

env = Multi_env.Environ(pos_2, pos_1, pos_3, pos_4, width, height, n_user, n_neighbor)
env.new_random_game()

n_episode = 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100

######################################################


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):


    UB_fast = (env.U2B_channels_with_fastfading[idx[0], :] - env.U2B_channels_abs[idx[0]] + 10)/35


    UU_fast = (env.U2U_channels_with_fastfading[:, env.users[idx[0]].destinations[idx[1]], :] - env.U2U_channels_abs[:, env.users[idx[0]].destinations[idx[1]]] + 10)/35

    UU_interference = (-env.U2U_Interference_all[idx[0], idx[1], :] - 60) / 60

    UB_abs = (env.U2B_channels_abs[idx[0]] - 80) / 60.0
    UU_abs = (env.U2U_channels_abs[:, env.users[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])


    return np.concatenate((UB_fast, np.reshape(UU_fast, -1), UU_interference, np.asarray([UB_abs]), UU_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


# -----------------------------------------------------------
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = n_RB * len(env.UU_power_List)

g = tf.Graph()
with g.as_default():

    x = tf.placeholder(tf.float32, [None, n_input])

    w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w_1), b_1))
    layer_1_b = tf.layers.batch_normalization(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = tf.layers.batch_normalization(layer_3)
    y = tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)


    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_loss)


    x_p = tf.placeholder(tf.float32, [None, n_input])

    w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1_p = tf.nn.relu(tf.add(tf.matmul(x_p, w_1_p), b_1_p))
    layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

    layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
    layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

    layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
    layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

    y_p = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p))

    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
    target_q_with_idx = tf.gather_nd(y_p, g_target_q_idx)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


def predict(sess, s_t, ep, test_ep = False):

    n_power_levels = len(env.UU_power_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB*n_power_levels)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action


def learning_mini_batch(current_agent, current_sess):


    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val


def update_target_network(sess):


    sess.run(w_1_p.assign(sess.run(w_1)))
    sess.run(w_2_p.assign(sess.run(w_2)))
    sess.run(w_3_p.assign(sess.run(w_3)))
    sess.run(w_4_p.assign(sess.run(w_4)))

    sess.run(b_1_p.assign(sess.run(b_1)))
    sess.run(b_2_p.assign(sess.run(b_2)))
    sess.run(b_3_p.assign(sess.run(b_3)))
    sess.run(b_4_p.assign(sess.run(b_4)))


def save_models(sess, model_path):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)


def load_models(sess, model_path):

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):


    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))


# --------------------------------------------------------------
print("Initializing agent...")
agent = Agent(memory_entry_size=len(get_state(env)))

sess = tf.Session(graph=g,config=my_config)
sess.run(init)

# -------------------------训练-----------------------------
if is_train:
    record_reward = np.zeros([n_episode * n_step_per_episode, 1])
    record_loss = []
    action_all_training = np.zeros([n_user, n_neighbor, 2], dtype='int32')
    time_step = 0
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)
        else:
            epsi = epsi_final
        if i_episode%100 == 0:
            env.renew_positions()
            env.renew_neighbor()
            env.renew_channel()
            env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_User, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_User, env.n_neighbor))
        env.active_links = np.ones((env.n_User, env.n_neighbor), dtype='bool')

        for i_step in range(n_step_per_episode):
            time_step = i_episode * n_step_per_episode + i_step

            remainder = i_step % (n_user * n_neighbor)
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor
            state = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
            action = predict(sess, state, epsi)
            action_all_training[i, j, 0] = action % n_RB
            action_all_training[i, j, 1] = int(np.floor(action / n_RB))

            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)
            record_reward[time_step] = train_reward

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
            agent.memory.add(state, state_new, train_reward, action)

            if time_step % mini_batch_step == mini_batch_step - 1:
                loss_val_batch = learning_mini_batch(agent, sess)
                record_loss.append(loss_val_batch)
                print('step:', time_step, 'loss', loss_val_batch)
            if time_step % target_update_step == target_update_step - 1:
                update_target_network(sess)
                print('Update target Q network...')

    print('Training Done. Saving models...')
    model_path = label + '/agent'
    save_models(sess, model_path)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')
    scipy.io.savemat(reward_path, {'reward': record_reward})

    record_loss = np.asarray(record_loss)
    loss_path = os.path.join(current_dir, "model/" + label + '/train_loss.mat')
    scipy.io.savemat(loss_path, {'train_loss': record_loss})


# -------------- 测试 --------------
if is_test:
    print("\nRestoring the model...")
    model_path = label + '/agent'
    load_models(sess, model_path)

    BU_rate_list = []
    UU_success_list = []
    BU_rate_list_rand = []
    UU_success_list_rand = []

    action_all_testing = np.zeros([n_user, n_neighbor, 2], dtype='int32')

    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_User, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_User, env.n_neighbor))
        env.active_links = np.ones((env.n_User, env.n_neighbor), dtype='bool')

        env.demand_rand = env.demand_size * np.ones((env.n_User, env.n_neighbor))
        env.individual_time_limit_rand = env.time_slow * np.ones((env.n_User, env.n_neighbor))
        env.active_links_rand = np.ones((env.n_User, env.n_neighbor), dtype='bool')

        BU_rate_per_episode = []
        BU_rate_per_episode_rand = []
        for test_step in range(n_step_per_episode):

            remainder = test_step % (n_user * n_neighbor)
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor
            state_old = get_state(env, [i, j], 1, epsi_final)
            action = predict(sess, state_old, epsi_final, True)
            action_all_testing[i, j, 0] = action % n_RB
            action_all_testing[i, j, 1] = int(np.floor(action / n_RB))

            action_temp = action_all_testing.copy()
            BU_rate, UU_success, UU_rate = env.act_for_testing(action_temp)
            BU_rate_per_episode.append(np.sum(BU_rate))

            action_rand = np.zeros([n_user, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_user, n_neighbor])
            action_rand[:, :, 1] = np.random.randint(0, len(env.UU_power_List), [n_user, n_neighbor])

            BU_rate_rand, UU_success_rand, UU_rate_rand = env.act_for_testing_rand(action_rand)
            BU_rate_per_episode_rand.append(np.sum(BU_rate_rand))

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            if test_step == int(n_step_per_episode / (n_user * n_neighbor)) - 1:
                UU_success_list.append(UU_success)
                UU_success_list_rand.append(UU_success_rand)

        BU_rate_list.append(np.mean(BU_rate_per_episode))
        BU_rate_list_rand.append(np.mean(BU_rate_per_episode_rand))

        print(round(np.average(BU_rate_per_episode), 2), 'rand', round(np.average(BU_rate_per_episode_rand), 2))
        print(UU_success_list[idx_episode], 'rand', UU_success_list_rand[idx_episode])

    print('-------- sarl -------------')
    print('n_user:', n_user, ', n_neighbor:', n_neighbor)
    print('Sum BU rate:', round(np.average(BU_rate_list), 2), 'Mbps')
    print('Pr(UU success):', round(np.average(UU_success_list), 4))

    print('-------- random -------------')
    print('n_user:', n_user, ', n_neighbor:', n_neighbor)
    print('Sum BU rate:', round(np.average(BU_rate_list_rand), 2), 'Mbps')
    print('Pr(UU success):', round(np.average(UU_success_list_rand), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- sarl, ' + label + '------\n')
        f.write('n_user: ' + str(n_user) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum BU rate: ' + str(round(np.average(BU_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(UU): ' + str(round(np.average(UU_success_list), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Sum BU rate: ' + str(round(np.average(BU_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Pr(UU): ' + str(round(np.average(UU_success_list_rand), 5)) + '\n')


sess.close()


