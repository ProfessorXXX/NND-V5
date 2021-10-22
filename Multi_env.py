from __future__ import division
import numpy as np
import time
import random
import math
from User import User
from usercom import UU
from base_station import BU

#np.random.seed(1234)

class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_user, n_neighbor):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.U2UCHANNELS = UU()
        self.U2BCHANNELS = BU()
        self.users = []

        self.demand = []
        self.U2U_Shadow = []
        self.U2B_Shadow = []
        self.delta_distance = []
        self.U2U_channels_abs = []
        self.U2B_channels_abs = []

        self.UB_power = 23  # dBm
        self.UU_power_List = [23, 15, 5, -100]  # power level
        self.sig2_dB = -110
        self.bsAntGain = 80
        self.bsNoiseFigure = 50
        self.UserAntGain = 50
        self.UserNoiseFigure = 80
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.n_RB = n_user
        self.n_User = n_user
        self.n_neighbor = n_neighbor
        self.time_fast = 0.001
        self.time_slow = 0.1  #
        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz
        self.demand_size = int(1024 * 8 * 2)  #


        self.U2U_Interference_all = np.zeros((self.n_User, self.n_neighbor, self.n_RB)) + self.sig2

    def add_new_users(self, start_position, start_dt, start_speed):
        self.users.append(User(start_position, start_dt, start_speed))


    def add_new_users_by_number(self, n):

        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_dt = 'd' # speed: 10 ~ 15 m/s, random
            self.add_new_users(start_position, start_dt, np.random.randint(10, 15))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_dt = 'u'
            self.add_new_users(start_position, start_dt, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_dt = 'l'
            self.add_new_users(start_position, start_dt, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_dt = 'r'
            self.add_new_users(start_position, start_dt, np.random.randint(10, 15))

        #
        self.U2U_Shadow = np.random.normal(0, 3, [len(self.users), len(self.users)])
        self.U2B_Shadow = np.random.normal(0, 8, len(self.users))
        self.delta_distance = np.asarray([c.speed * self.time_slow for c in self.users])

    def renew_positions(self):


        i = 0
        while (i < len(self.users)):
            delta_distance = self.users[i].speed * self.time_slow
            change_dt = False
            if self.users[i].dt == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.users[i].position[1] <= self.left_lanes[j]) and ((self.users[i].position[1] + delta_distance) >= self.left_lanes[j]):  #
                        if (np.random.uniform(0, 1) < 0.4):
                            self.users[i].position = [self.users[i].position[0] - (delta_distance - (self.left_lanes[j] - self.users[i].position[1])), self.left_lanes[j]]
                            self.users[i].dt = 'l'
                            change_dt = True
                            break
                if change_dt == False:
                    for j in range(len(self.right_lanes)):
                        if (self.users[i].position[1] <= self.right_lanes[j]) and ((self.users[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.users[i].position = [self.users[i].position[0] + (delta_distance + (self.right_lanes[j] - self.users[i].position[1])), self.right_lanes[j]]
                                self.users[i].dt = 'r'
                                change_dt = True
                                break
                if change_dt == False:
                    self.users[i].position[1] += delta_distance
            if (self.users[i].dt == 'd') and (change_dt == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.users[i].position[1] >= self.left_lanes[j]) and ((self.users[i].position[1] - delta_distance) <= self.left_lanes[j]):  # 
                        if (np.random.uniform(0, 1) < 0.4):
                            self.users[i].position = [self.users[i].position[0] - (delta_distance - (self.users[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.users[i].position)
                            self.users[i].dt = 'l'
                            change_dt = True
                            break
                if change_dt == False:
                    for j in range(len(self.right_lanes)):
                        if (self.users[i].position[1] >= self.right_lanes[j]) and (self.users[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.users[i].position = [self.users[i].position[0] + (delta_distance + (self.users[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.users[i].position)
                                self.users[i].dt = 'r'
                                change_dt = True
                                break
                if change_dt == False:
                    self.users[i].position[1] -= delta_distance
            if (self.users[i].dt == 'r') and (change_dt == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.users[i].position[0] <= self.up_lanes[j]) and ((self.users[i].position[0] + delta_distance) >= self.up_lanes[j]):
                        if (np.random.uniform(0, 1) < 0.4):
                            self.users[i].position = [self.up_lanes[j], self.users[i].position[1] + (delta_distance - (self.up_lanes[j] - self.users[i].position[0]))]
                            change_dt = True
                            self.users[i].dt = 'u'
                            break
                if change_dt == False:
                    for j in range(len(self.down_lanes)):
                        if (self.users[i].position[0] <= self.down_lanes[j]) and ((self.users[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.users[i].position = [self.down_lanes[j], self.users[i].position[1] - (delta_distance - (self.down_lanes[j] - self.users[i].position[0]))]
                                change_dt = True
                                self.users[i].dt = 'd'
                                break
                if change_dt == False:
                    self.users[i].position[0] += delta_distance
            if (self.users[i].dt == 'l') and (change_dt == False):
                for j in range(len(self.up_lanes)):

                    if (self.users[i].position[0] >= self.up_lanes[j]) and ((self.users[i].position[0] - delta_distance) <= self.up_lanes[j]):
                        if (np.random.uniform(0, 1) < 0.4):
                            self.users[i].position = [self.up_lanes[j], self.users[i].position[1] + (delta_distance - (self.users[i].position[0] - self.up_lanes[j]))]
                            change_dt = True
                            self.users[i].dt = 'u'
                            break
                if change_dt == False:
                    for j in range(len(self.down_lanes)):
                        if (self.users[i].position[0] >= self.down_lanes[j]) and ((self.users[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.users[i].position = [self.down_lanes[j], self.users[i].position[1] - (delta_distance - (self.users[i].position[0] - self.down_lanes[j]))]
                                change_dt = True
                                self.users[i].dt = 'd'
                                break
                    if change_dt == False:
                        self.users[i].position[0] -= delta_distance

            # 如果用户移动远离某个范围就退出
            if (self.users[i].position[0] < 0) or (self.users[i].position[1] < 0) or (self.users[i].position[0] > self.width) or (self.users[i].position[1] > self.height):

                #    print ('delete ', self.position[i])
                if (self.users[i].dt == 'u'):
                    self.users[i].dt = 'r'
                    self.users[i].position = [self.users[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.users[i].dt == 'd'):
                        self.users[i].dt = 'l'
                        self.users[i].position = [self.users[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.users[i].dt == 'l'):
                            self.users[i].dt = 'u'
                            self.users[i].position = [self.up_lanes[0], self.users[i].position[1]]
                        else:
                            if (self.users[i].dt == 'r'):
                                self.users[i].dt = 'd'
                                self.users[i].position = [self.down_lanes[-1], self.users[i].position[1]]

            i += 1

    def renew_neighbor(self):

        for i in range(len(self.users)):
            self.users[i].neighbors = []
            self.users[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.users]])
        Distance = abs(z.T - z)

        for i in range(len(self.users)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(self.n_neighbor):
                self.users[i].neighbors.append(sort_idx[j + 1])
            destination = self.users[i].neighbors

            self.users[i].destinations = destination

    def renew_channel(self):


        self.U2U_PL = np.zeros((len(self.users), len(self.users))) + 50 * np.identity(len(self.users))
        self.U2B_PL = np.zeros((len(self.users)))

        self.U2U_channels_abs = np.zeros((len(self.users), len(self.users)))
        self.U2B_channels_abs = np.zeros((len(self.users)))
        for i in range(len(self.users)):
            for j in range(i + 1, len(self.users)):
                self.U2U_Shadow[j][i] = self.U2U_Shadow[i][j] = self.U2UCHANNELS.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.U2U_Shadow[i][j])
                self.U2U_PL[j, i] = self.U2U_PL[i][j] = self.U2UCHANNELS.get_PL(self.users[i].position, self.users[j].position)

        self.U2U_channels_abs = self.U2U_PL + self.U2U_Shadow

        self.U2B_Shadow = self.U2BCHANNELS .get_shadowing(self.delta_distance, self.U2B_Shadow)
        for i in range(len(self.users)):
            self.U2B_PL[i] = self.U2BCHANNELS.get_PL(self.users[i].position)

        self.U2B_channels_abs = self.U2B_PL + self.U2B_Shadow

    def renew_channels_fastfading(self):


        U2U_channels_with_fastfading = np.repeat(self.U2U_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.U2U_channels_with_fastfading = U2U_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, U2U_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, U2U_channels_with_fastfading.shape)) / math.sqrt(2))

        U2B_channels_with_fastfading = np.repeat(self.U2B_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.U2B_channels_with_fastfading = U2B_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, U2B_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, U2B_channels_with_fastfading.shape))/ math.sqrt(2))

    def Compute_Performance_Reward_Train(self, actions_power):

        actions = actions_power[:, :, 0]  #
        power_selection = actions_power[:, :, 1]  #

        # ------------rate --------------------
        UU_Rate = np.zeros(self.n_RB)
        UB_Interference = np.zeros(self.n_RB)  #
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.U2B_channels_with_fastfading[i, actions[i, j]]
                                                           + self.UserAntGain + self.bsAntGain - self.bsNoiseFigure)/10 )
        self.UB_Interference = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.U2B_channels_with_fastfading.diagonal() + self.UserAntGain + self.bsAntGain - self.bsNoiseFigure)/10 )
        UU_Rate = np.log2(1 + np.divide(UB_Signals, self.UB_Interference))

        # ------------ rate -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links))] = -1 # inactive
        for i in range(self.n_RB):  #
            indexes = np.argwhere(actions == i)  #
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.U2U_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.UserAntGain - self.UserNoiseFigure) /10 )
                #
                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.U2U_channels_with_fastfading[i, receiver_j, i] + 2 * self.UserAntGain - self.UserNoiseFigure) /10 )

                #
                for k in range(j + 1, len(indexes)):  #
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.U2U_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.UserAntGain - self.UserNoiseFigure)/10 )
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.U2U_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.UserAntGain - self.UserNoiseFigure)/10 )
        self.UU_Interference = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, self.UU_Interference))

        self.demand -= UU_Rate * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0 #

        self.individual_time_limit -= self.time_fast

        reward_elements = UU_Rate/10
        reward_elements[self.demand <= 0] = 1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0 #

        return UU_Rate, UU_Rate, reward_elements

    def Compute_Performance_Reward_Test_rand(self, actions_power):


        actions = actions_power[:, :, 0]  #
        power_selection = actions_power[:, :, 1]  #

        # ------------ rate --------------------
        UB_Rate = np.zeros(self.n_RB)
        UB_Interference = np.zeros(self.n_RB)  #
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links_rand[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.U2B_channels_with_fastfading[i, actions[i, j]]
                                                           + self.UserAntGain + self.bsAntGain - self.bsNoiseFigure) /10 )
        self.UB_Interference_random = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.U2B_channels_with_fastfading.diagonal() + self.UserAntGain + self.bsAntGain - self.bsNoiseFigure) /10 )
        UB_Rate = np.log2(1 + np.divide(UB_Signals, self.UB_Interference_random))

        # ------------ rate -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):  #
            indexes = np.argwhere(actions == i)  #
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.U2U_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.UserAntGain - self.UserNoiseFigure)/10 )
                #
                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.U2U_channels_with_fastfading[i, receiver_j, i] + 2 * self.UserAntGain - self.UserNoiseFigure) /10 )

                #
                for k in range(j + 1, len(indexes)):  #
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.U2U_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.UserAntGain - self.UserNoiseFigure)/10 )
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.U2U_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.UserAntGain - self.UserNoiseFigure)/10 )
        self.UU_Interference_random = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, self.UU_Interference_random))

        self.demand_rand -= UU_Rate * self.time_fast * self.bandwidth
        self.demand_rand[self.demand_rand < 0] = 0

        self.individual_time_limit_rand -= self.time_fast

        self.active_links_rand[np.multiply(self.active_links_rand, self.demand_rand <= 0)] = 0 #

        return UB_Rate, UU_Rate

    def Compute_Interference(self, actions):
        UU_Interference = np.zeros((len(self.users), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        #
        for i in range(self.n_RB):
            for k in range(len(self.users)):
                for m in range(len(channel_selection[k, :])):
                    UU_Interference[k, m, i] += 10 ** ((self.UB_power - self.U2U_channels_with_fastfading[i][self.users[k].destinations[m]][i] + 2 * self.UserAntGain - self.UserNoiseFigure)/10 )

        #
        for i in range(len(self.users)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.users)):
                    for m in range(len(channel_selection[k, :])):
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        UU_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.UU_power_List[power_selection[i, j]]
                                                                                   - self.U2U_channels_with_fastfading[i][self.users[k].destinations[m]][channel_selection[i, j]] + 2 * self.UserAntGain - self.UserNoiseFigure) /10)
        self.U2U_Interference_all = 10 * np.log10(UU_Interference)


    def act_for_training(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)


        lambdda = 0.5
        reward = lambdda * np.sum(UB_Rate) / (self.n_User * 10) + (1 - lambdda) * np.sum(reward_elements) / (self.n_User * self.n_neighbor)


        return reward

    def act_for_testing(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
        UU_success = 1 - np.sum(self.active_links) / (self.n_User * self.n_neighbor)  # UU success rates

        return UB_Rate, UU_success, UU_Rate

    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
        UU_success = 1 - np.sum(self.active_links_rand) / (self.n_User * self.n_neighbor)  # V2V success rates

        return UB_Rate, UU_success, UU_Rate

    def new_random_game(self, n_User=0):
        #

        self.users = []
        if n_User > 0:
            self.n_User = n_User
        self.add_new_users_by_number(int(self.n_User / 4))
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()

        self.demand = self.demand_size * np.ones((self.n_User, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_User, self.n_neighbor))
        self.active_links = np.ones((self.n_User, self.n_neighbor), dtype='bool')

        # random baseline
        self.demand_rand = self.demand_size * np.ones((self.n_User, self.n_neighbor))
        self.individual_time_limit_rand = self.time_slow * np.ones((self.n_User, self.n_neighbor))
        self.active_links_rand = np.ones((self.n_User, self.n_neighbor), dtype='bool')



