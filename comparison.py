from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Multi_env_comparison
import os
from pool import ReplayMemory
import sys

my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True # True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################
pos_1 = [i for i in [3.5, 3.5+ 3.5, 250 + 3.5, 250 + 3.5 + 3.5, 500 + 3.5, 500 + 3.5 + 3.5]]
pos_2 = [i for i in [250 - 3.5 - 3.5, 250 - 3.5, 500 - 3.5 - 3.5, 500 - 3.5, 750 - 3.5 - 3.5, 750 - 3.5]]
pos_3 = [i for i in [3.5, 3.5+ 3.5, 433 + 3.5, 433 + 3.5 + 3.5, 866 + 3.5, 866 + 3.5 + 3.5]]
pos_4 = [i for i in [433 - 3.5 - 3.5, 433 - 3.5, 866 - 3.5 - 3.5, 866 - 3.5, 1299 - 3.5 - 3.5, 1299 - 3.5]]

width = 750
height = 1298


is_train = 0
is_test = 1 - is_train

label = 'multi'
label_single = 'single'

n_user = 4
n_neighbor = 1
n_RB = n_user # Resource Block

env = Multi_env_comparison.Environ(pos_2, pos_1, pos_3, pos_4, width, height, n_user, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 2500
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

######################################################


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):

    U2B_fast = (env.UB_channels_fastfading[idx[0], :] - env.UB_channels_abs[idx[0]] + 10)/35

    U2U_fast = (env.UU_channels_with_fastfading[:, env.users[idx[0]].destinations[idx[1]], :] - env.UU_channels_abs[:, env.users[idx[0]].destinations[idx[1]]] + 10)/35

    U2U_interference = (-env.UU_Interference_all[idx[0], idx[1], :] - 60) / 60

    U2B_abs = (env.UB_channels_abs[idx[0]] - 80) / 60.0
    U2U_abs = (env.UU_channels_abs[:, env.users[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((U2B_fast, np.reshape(U2U_fast, -1), U2U_interference, np.asarray([U2B_abs]), U2U_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


def get_state_sarl(env, idx=(0,0), ind_episode=1., epsi=0.02):

    U2B_fast = (env.UB_channels_fastfading[idx[0], :] - env.UB_channels_abs[idx[0]] + 10)/35

    U2U_fast = (env.UU_channels_with_fastfading[:, env.users[idx[0]].destinations[idx[1]], :] - env.UU_channels_abs[:, env.users[idx[0]].destinations[idx[1]]] + 10)/35

    U2U_interference = (-env.UU_Interference_all_single[idx[0], idx[1], :] - 60) / 60

    U2B_abs = (env.UB_channels_abs[idx[0]] - 80) / 60.0
    U2U_abs = (env.UU_channels_abs[:, env.users[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand_sarl[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit_sarl[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((U2B_fast, np.reshape(U2U_fast, -1), U2U_interference, np.asarray([U2B_abs]), U2U_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


# -----------------------------------------------------------
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = n_RB * len(env.UU_power_List)

g = tf.Graph()
with g.as_default():
    # ============== train net ========================
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

    # loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_loss)

    # ==================== predict ========================
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

def predict_sarl(sess, s_t):
    pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action


def q_learning_mini_batch(current_agent, current_sess):

    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:  #
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val


def update_target_q_network(sess):

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
    model_path = os.path.join(current_dir, "comparison_model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)


def load_models(sess, model_path):


    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "comparison_model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):


    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))


# --------------------------------------------------------------
agents = []
sesses = []
for ind_agent in range(n_user * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    sess.run(init)
    sesses.append(sess)

agent_sarl = Agent(memory_entry_size=len(get_state(env)))
sess_sarl = tf.Session(graph=g,config=my_config)
sess_sarl.run(init)

# -------------- Testing --------------
if is_test:
    print("\nRestoring the model...")

    for i in range(n_user):
        for j in range(n_neighbor):
            model_path = label + '/agent_' + str(i * n_neighbor + j)
            load_models(sesses[i * n_neighbor + j], model_path)
    #
    model_path_single = label_single + '/agent'
    load_models(sess_sarl, model_path_single)

    BU_rate_list = []
    UU_rate_list = []
    UU_success_list = []

    BU_rate_list_rand = []
    UU_rate_list_rand = []
    UU_success_list_rand = []

    BU_rate_list_sarl = []
    UU_rate_list_sarl = []

    UU_success_list_sarl = []

    UB_rate_list_dpra = []
    UU_rate_list_dpra = []
    UU_success_list_dpra = []

    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_user, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_user, n_neighbor])
    demand_marl = env.demand_size * np.ones([n_episode_test, n_step_per_episode + 1, n_user, n_neighbor])
    demand_rand = env.demand_size * np.ones([n_episode_test, n_step_per_episode + 1, n_user, n_neighbor])
    demand_sarl = env.demand_size * np.ones([n_episode_test, n_step_per_episode + 1, n_user, n_neighbor])
    demand_dpra = env.demand_size * np.ones([n_episode_test, n_step_per_episode + 1, n_user, n_neighbor])

    action_all_testing_sarl = np.zeros([n_user, n_neighbor, 2], dtype='int32')
    action_all_testing_dpra = np.zeros([n_user, n_neighbor, 2], dtype='int32')
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

        env.demand_sarl = env.demand_size * np.ones((env.n_User, env.n_neighbor))
        env.individual_time_limit_sarl = env.time_slow * np.ones((env.n_User, env.n_neighbor))
        env.active_links_sarl = np.ones((env.n_User, env.n_neighbor), dtype='bool')

        env.demand_dpra = env.demand_size * np.ones((env.n_User, env.n_neighbor))
        env.individual_time_limit_dpra = env.time_slow * np.ones((env.n_User, env.n_neighbor))
        env.active_links_dpra = np.ones((env.n_User, env.n_neighbor), dtype='bool')

        BU_rate_per_episode = []
        UU_rate_per_episode = []

        BU_rate_per_episode_rand = []
        UU_rate_per_episode_rand = []

        BU_rate_per_episode_sarl = []
        UU_rate_per_episode_sarl = []
        BU_rate_per_episode_dpra = []
        UU_rate_per_episode_dpra = []

        for test_step in range(n_step_per_episode):
            # trained models
            action_all_testing = np.zeros([n_user, n_neighbor, 2], dtype='int32')
            for i in range(n_user):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action = predict(sesses[i*n_neighbor+j], state_old, epsi_final, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen 信道资源
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # 功率

            action_temp = action_all_testing.copy()
            BU_rate, UU_success, UU_rate = env.act_for_testing(action_temp)
            BU_rate_per_episode.append(np.sum(BU_rate))  # sum UB rate / bps
            UU_rate_per_episode.append(np.sum(UU_rate))


            rate_marl[idx_episode, test_step,:,:] = UU_rate
            demand_marl[idx_episode, test_step+1,:,:] = env.demand

            # random baseline
            action_rand = np.zeros([n_user, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_user, n_neighbor]) # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.UU_power_List), [n_user, n_neighbor]) # 功率
            UB_rate_rand, UU_success_rand, UU_rate_rand = env.act_for_testing_rand(action_rand)
            BU_rate_per_episode_rand.append(np.sum(UB_rate_rand))  # sum UB rate / bps
            UU_rate_per_episode_rand.append(np.sum(UU_rate_rand))

            rate_rand[idx_episode, test_step, :, :] = UU_rate_rand
            demand_rand[idx_episode, test_step+1,:,:] = env.demand_rand

            # single RL
            remainder = test_step % (n_user * n_neighbor)
            i = int(np.floor(remainder/n_neighbor))
            j = remainder % n_neighbor
            state_sarl = get_state_sarl(env, [i, j], 1, epsi_final)
            action = predict_sarl(sess_sarl, state_sarl)
            action_all_testing_sarl[i, j, 0] = action % n_RB  # chosen RB
            action_all_testing_sarl[i, j, 1] = int(np.floor(action / n_RB))  # power level
            action_temp_sarl = action_all_testing_sarl.copy()
            BU_rate_single, UU_success_single, UU_rate_single = env.act_for_testing_sarl(action_temp_sarl)
            BU_rate_per_episode_sarl.append(np.sum(BU_rate_single))  # sum UB rate in bps
            UU_rate_per_episode_sarl.append(np.sum(UU_rate_single))

            demand_sarl[idx_episode, test_step + 1, :, :] = env.demand_sarl




            action_dpra = np.zeros([n_user, n_neighbor, 2], dtype='int32')
            n_power_level = 1
            store_action = np.zeros([(n_RB*n_power_level)**4, 4])
            rate_all_dpra = []
            t = 0

            for i in range(n_RB):
                for j in range(n_RB):
                    for m in range(n_RB):
                        for n in range(n_RB):
                            action_dpra[0, 0, 0] = i % n_RB
                            action_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level

                            action_dpra[1, 0, 0] = j % n_RB
                            action_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level

                            action_dpra[2, 0, 0] = m % n_RB
                            action_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level

                            action_dpra[3, 0, 0] = n % n_RB
                            action_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level

                            action_temp_findMax = action_dpra.copy()
                            BU_rate_findMax, UU_rate_findMax = env.Compute_Rate(action_temp_findMax)
                            rate_all_dpra.append(np.sum(UU_rate_findMax))

                            store_action[t, :] = [i,j,m,n]
                            t += 1

            i = store_action[np.argmax(rate_all_dpra), 0]
            j = store_action[np.argmax(rate_all_dpra), 1]
            m = store_action[np.argmax(rate_all_dpra), 2]
            n = store_action[np.argmax(rate_all_dpra), 3]

            action_testing_dpra = np.zeros([n_user, n_neighbor, 2], dtype='int32')

            action_testing_dpra[0, 0, 0] = i % n_RB
            action_testing_dpra[0, 0, 1] = int(np.floor(i / n_RB))  # power level

            action_testing_dpra[1, 0, 0] = j % n_RB
            action_testing_dpra[1, 0, 1] = int(np.floor(j / n_RB))  # power level

            action_testing_dpra[2, 0, 0] = m % n_RB
            action_testing_dpra[2, 0, 1] = int(np.floor(m / n_RB))  # power level

            action_testing_dpra[3, 0, 0] = n % n_RB
            action_testing_dpra[3, 0, 1] = int(np.floor(n / n_RB))  # power level

            BU_rate_findMax, UU_rate_findMax = env.Compute_Rate(action_testing_dpra)
            check_sum = np.sum(UU_rate_findMax)

            demand_dpra[idx_episode, test_step + 1, :, :] = env.demand_dpra

            action_temp_dpra = action_testing_dpra.copy()
            BU_rate_dpra, UU_success_dpra, UU_rate_dpra = env.act_for_testing_dpra(action_temp_dpra)
            BU_rate_per_episode_dpra.append(np.sum(BU_rate_dpra))  # sum UB rate / bps
            UU_rate_per_episode_dpra.append(np.sum(UU_rate_dpra))
            #
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            env.Compute_Interference_sarl(action_temp_sarl)
            env.Compute_Interference_dpra(action_temp_dpra)

            if test_step == n_step_per_episode - 1:
                UU_success_list.append(UU_success)
                UU_success_list_rand.append(UU_success_rand)
                UU_success_list_sarl.append(UU_success_single)
                UU_success_list_dpra.append(UU_success_dpra)

        BU_rate_list.append(np.mean(BU_rate_per_episode))
        UU_rate_list.append(np.mean(UU_rate_per_episode))

        BU_rate_list_rand.append(np.mean(BU_rate_per_episode_rand))
        UU_rate_list_rand.append(np.mean(UU_rate_per_episode_rand))

        BU_rate_list_sarl.append(np.mean(BU_rate_per_episode_sarl))
        UU_rate_list_sarl.append(np.mean(UU_rate_per_episode_sarl))

        UB_rate_list_dpra.append(np.mean(BU_rate_per_episode_dpra))
        UU_rate_list_dpra.append(np.mean(UU_rate_per_episode_dpra))
        print('BU marl', round(np.average(BU_rate_per_episode), 2),'UU marl', round(np.average(UU_rate_per_episode), 2), 'BU sarl', round(np.average(BU_rate_per_episode_sarl), 2), 'UU sarl', round(np.average(UU_rate_per_episode_sarl), 2), 'BU rand', round(np.average(BU_rate_per_episode_rand), 2), 'UU rand',round(np.average(UU_rate_per_episode_rand), 2), 'BU DMARL', round(np.average(BU_rate_per_episode_dpra), 2),'UU DMARL', round(np.average(UU_rate_per_episode_dpra), 2))
        print('marl', UU_success_list[idx_episode], 'sarl', UU_success_list_sarl[idx_episode], 'rand', UU_success_list_rand[idx_episode], 'DMARL', UU_success_list_dpra[idx_episode])

    print('-------- marl -------------')
    print('n_user:', n_user, ', n_neighbor:', n_neighbor)
    print('Sum BU rate:', round(np.average(BU_rate_list), 2), 'Mbps')
    print('Sum UU rate:', round(np.average(UU_rate_list), 2), 'Mbps')
    print('Pr(UU success):', round(np.average(UU_success_list), 4))
    #
    print('-------- sarl -------------')
    print('n_user:', n_user, ', n_neighbor:', n_neighbor)
    print('Sum BU rate:', round(np.average(BU_rate_list_sarl), 2), 'Mbps')
    print('Sum UU rate:', round(np.average(UU_rate_list_sarl), 2), 'Mbps')
    print('Pr(UU success):', round(np.average(UU_success_list_sarl), 4))

    print('-------- random -------------')
    print('n_user:', n_user, ', n_neighbor:', n_neighbor)
    print('Sum BU rate:', round(np.average(BU_rate_list_rand), 2), 'Mbps')
    print('Sum UU rate:', round(np.average(UU_rate_list_rand), 2), 'Mbps')
    print('Pr(UU success):', round(np.average(UU_success_list_rand), 4))

    print('-------- DMARL -------------')
    print('n_user:', n_user, ', n_neighbor:', n_neighbor)
    print('Sum BU rate:', round(np.average(UB_rate_list_dpra), 2), 'Mbps')
    print('Sum UU rate:', round(np.average(UU_rate_list_dpra), 2), 'Mbps')
    print('Pr(UU success):', round(np.average(UU_success_list_dpra), 4))



    with open("Data_com.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('n_user: ' + str(n_user) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum BU rate: ' + str(round(np.average(BU_rate_list), 5)) + ' Mbps\n')
        f.write('Sum UU rate: ' + str(round(np.average(UU_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(UU): ' + str(round(np.average(UU_success_list), 5)) + '\n')
        f.write('-------- sarl, ' + label_single + '------\n')
        f.write('n_user: ' + str(n_user) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum BU rate: ' + str(round(np.average(BU_rate_list_sarl), 5)) + ' Mbps\n')
        f.write('Sum UU rate: ' + str(round(np.average(UU_rate_list_sarl), 5)) + ' Mbps\n')
        f.write('Pr(UU): ' + str(round(np.average(UU_success_list_sarl), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Sum BU rate: ' + str(round(np.average(BU_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Sum UU rate: ' + str(round(np.average(UU_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Pr(UU): ' + str(round(np.average(UU_success_list_rand), 5)) + '\n')
        f.write('--------DMARL ------------\n')
        f.write('Dpra Sum BU rate: ' + str(round(np.average(UB_rate_list_dpra), 5)) + ' Mbps\n')
        f.write('Dpra Sum UU rate: ' + str(round(np.average(UU_rate_list_dpra), 5)) + ' Mbps\n')
        f.write('Dpra Pr(UU): ' + str(round(np.average(UU_success_list_dpra), 5)) + '\n')

    current_dir = os.path.dirname(os.path.realpath(__file__))
    marl_path = os.path.join(current_dir, "comparison_model/" + label + '/rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    rand_path = os.path.join(current_dir, "comparison_model/" + label + '/rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

    demand_marl_path = os.path.join(current_dir, "comparison_model/" + label + '/demand_marl.mat')
    scipy.io.savemat(demand_marl_path, {'demand_marl': demand_marl})
    demand_rand_path = os.path.join(current_dir, "comparison_model/" + label + '/demand_rand.mat')
    scipy.io.savemat(demand_rand_path, {'demand_rand': demand_rand})

    demand_sarl_path = os.path.join(current_dir, "comparison_model/" + label + '/demand_sarl.mat')
    scipy.io.savemat(demand_sarl_path, {'demand_rand': demand_sarl})

    demand_dpra_path = os.path.join(current_dir, "comparison_model/" + label + '/demand_dpra.mat')
    scipy.io.savemat(demand_dpra_path, {'demand_rand': demand_dpra})

for sess in sesses:
    sess.close()

