from __future__ import division
import numpy as np
import time
import random
import math
from usercom import UU
from base_station import BU
from User import User
np.random.seed(1234)

class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_user, n_neighbor):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.UUchannels = UU()
        self.UBchannels = BU()
        self.users = []

        self.demand = []
        self.UU_Shadow = []
        self.UB_Shadow = []
        self.delta_distance = []
        self.UU_channels_abs = []
        self.UB_channels_abs = []

        self.UB_power = 23  # dBm
        self.UU_power_List = [23, 15, 5, 0]  # the power levels
        self.sig2_dB = -114
        self.bsAntGain = 80
        self.bsNoiseFigure = 50
        self.userAntGain = 30
        self.userNoiseFigure = 90
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.n_RB = n_user
        self.n_User = n_user
        self.n_neighbor = n_neighbor
        self.time_fast = 0.001
        self.time_slow = 0.1  # 100 ms
        self.bandwidth = 10 * int(1e6)  # bandwidth per RB, 1 MHz
        self.demand_size = int(1024 * 10)  #  1024 Bytes / 100 ms


        self.UU_Interference_all = np.zeros((self.n_User, self.n_neighbor, self.n_RB)) + self.sig2
        self.UU_Interference_all_single = np.zeros((self.n_User, self.n_neighbor, self.n_RB)) + self.sig2
        self.UU_Interference_all_dpra = np.zeros((self.n_User, self.n_neighbor, self.n_RB)) + self.sig2

    def add_new_users(self, start_position, start_dt, start_velocity):
        self.users.append(User(start_position, start_dt, start_velocity))

    def add_new_users_by_number(self, n):

        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_dt = 'd' # speed: 5 ~ 15 m/s, random
            self.add_new_users(start_position, start_dt, np.random.randint(5, 15))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_dt = 'u'
            self.add_new_users(start_position, start_dt, np.random.randint(5, 15))
            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_dt = 'l'
            self.add_new_users(start_position, start_dt, np.random.randint(5, 15))
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_dt = 'r'
            self.add_new_users(start_position, start_dt, np.random.randint(5, 15))

        # init channels
        self.UU_Shadow = np.random.normal(0, 3, [len(self.users), len(self.users)])
        self.UB_Shadow = np.random.normal(0, 8, len(self.users))
        self.delta_distance = np.asarray([c.speed * self.time_slow for c in self.users])

    def renew_positions(self):

        i = 0
        while (i < len(self.users)):
            delta_distance = self.users[i].speed * self.time_slow
            change_dt = False
            if self.users[i].dt == 'u':
                for j in range(len(self.left_lanes)):

                    if (self.users[i].position[1] <= self.left_lanes[j]) and ((self.users[i].position[1] + delta_distance) >= self.left_lanes[j]):
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
                for j in range(len(self.left_lanes)):
                    if (self.users[i].position[1] >= self.left_lanes[j]) and ((self.users[i].position[1] - delta_distance) <= self.left_lanes[j]):  #
                        if (np.random.uniform(0, 1) < 0.4):
                            self.users[i].position = [self.users[i].position[0] - (delta_distance - (self.users[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            self.users[i].dt = 'l'
                            change_dt = True
                            break
                if change_dt == False:
                    for j in range(len(self.right_lanes)):
                        if (self.users[i].position[1] >= self.right_lanes[j]) and (self.users[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.users[i].position = [self.users[i].position[0] + (delta_distance + (self.users[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                self.users[i].direction = 'r'
                                change_dt = True
                                break
                if change_dt == False:
                    self.users[i].position[1] -= delta_distance
            if (self.users[i].dt == 'r') and (change_dt == False):
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

            # EIXT
            if (self.users[i].position[0] < 0) or (self.users[i].position[1] < 0) or (self.users[i].position[0] > self.width) or (self.users[i].position[1] > self.height):

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


        self.UU_PL = np.zeros((len(self.users), len(self.users))) + 50 * np.identity(len(self.users))
        self.UB_PL = np.zeros((len(self.users)))

        self.UU_channels_abs = np.zeros((len(self.users), len(self.users)))
        self.UB_channels_abs = np.zeros((len(self.users)))
        for i in range(len(self.users)):
            for j in range(i + 1, len(self.users)):
                self.UU_Shadow[j][i] = self.UU_Shadow[i][j] = self.UUchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.UU_Shadow[i][j])
                self.UU_PL[j, i] = self.UU_PL[i][j] = self.UUchannels.get_PL(self.users[i].position, self.users[j].position)

        self.UU_channels_abs = self.UU_PL + self.UU_Shadow

        self.UB_Shadow = self.UBchannels.get_shadowing(self.delta_distance, self.UB_Shadow)
        for i in range(len(self.users)):
            self.UB_PL[i] = self.UBchannels.get_PL(self.users[i].position)

        self.UB_channels_abs = self.UB_PL + self.UB_Shadow

    def renew_channels_fastfading(self):

        UU_channels_with_fastfading = np.repeat(self.UU_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.UU_channels_with_fastfading = UU_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, UU_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, UU_channels_with_fastfading.shape)) / math.sqrt(2))

        UB_channels_with_fastfading = np.repeat(self.UB_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.UB_channels_fastfading = UB_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, UB_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, UB_channels_with_fastfading.shape))/ math.sqrt(2))

    def Compute_Performance_Reward_Train(self, actions_power):

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # --------------------------------
        UB_Rate = np.zeros(self.n_RB)
        UB_Interference = np.zeros(self.n_RB)  #
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.UB_channels_fastfading[i, actions[i, j]]
                                                           + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.UB_Interference = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.UB_channels_fastfading.diagonal() + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Rate = np.log2(1 + np.divide(UB_Signals, self.UB_Interference))

        # ------------  -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links))] = -1 #
        for i in range(self.n_RB):  #
            indexes = np.argwhere(actions == i)  #
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.UU_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
                #
                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i, receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

                #
                for k in range(j + 1, len(indexes)):  #
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        self.UU_Interference = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, self.UU_Interference))

        self.demand -= UU_Rate * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0 #

        self.individual_time_limit -= self.time_fast

        reward_elements = UU_Rate/10
        reward_elements[self.demand <= 0] = 1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0 #

        return UB_Rate, UU_Rate * 100, reward_elements

    def Compute_Performance_Reward_Test_rand(self, actions_power):


        actions = actions_power[:, :, 0]  #
        power_selection = actions_power[:, :, 1]  #

        # ------------  rate --------------------
        UB_Rate = np.zeros(self.n_RB)
        UB_Interference = np.zeros(self.n_RB)  #
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links_rand[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.UB_channels_fastfading[i, actions[i, j]]
                                                           + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.UB_Interference_random = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.UB_channels_fastfading.diagonal() + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Rate = np.log2(1 + np.divide(UB_Signals, self.UB_Interference_random))

        # ------------ rate -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.UU_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i, receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)


                for k in range(j + 1, len(indexes)):
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        self.UU_Interference_random = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, self.UU_Interference_random))

        self.demand_rand -= UU_Rate * self.time_fast * self.bandwidth
        self.demand_rand[self.demand_rand < 0] = 0

        self.individual_time_limit_rand -= self.time_fast
        self.active_links_rand[np.multiply(self.active_links_rand, self.demand_rand <= 0)] = 0 # transmission finished, turned to "inactive"

        return UB_Rate, UU_Rate

    def Compute_Performance_Reward_Test_sarl(self, actions_power):


        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]

        # ------------ CM1 rate --------------------
        UB_Rate = np.zeros(self.n_RB)
        UB_Interference = np.zeros(self.n_RB)  # UB interference
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links_sarl[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.UB_channels_fastfading[i, actions[i, j]]
                                                           + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Interference_single = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.UB_channels_fastfading.diagonal() + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Rate = np.log2(1 + np.divide(UB_Signals, UB_Interference_single))

        # ------------ CM2 rate -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links_sarl))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)  # z找到合适的channel
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.UU_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i, receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)


                for k in range(j + 1, len(indexes)):
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        UU_Interference_single = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, UU_Interference_single))

        self.demand_sarl -= UU_Rate * self.time_fast * self.bandwidth
        self.demand_sarl[self.demand_sarl < 0] = 0

        self.individual_time_limit_sarl -= self.time_fast
        self.active_links_sarl[np.multiply(self.active_links_sarl, self.demand_sarl <= 0)] = 0 # 传终

        return UB_Rate, UU_Rate * 100

    def Compute_Performance_Reward_Test_dpra(self, actions_power):


        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]

        # -----------CM1 rate --------------------
        UB_Rate = np.zeros(self.n_RB)
        UB_Interference = np.zeros(self.n_RB)
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links_dpra[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.UB_channels_fastfading[i, actions[i, j]]
                                                           + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.UB_Interference_dpra = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.UB_channels_fastfading.diagonal() + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Rate = np.log2(1 + np.divide(UB_Signals, self.UB_Interference_dpra))

        # ------------ CM2 rate -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links_dpra))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.UU_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i, receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)


                for k in range(j + 1, len(indexes)):
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        self.UU_Interference_dpra = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, self.UU_Interference_dpra))

        self.demand_dpra -= UU_Rate * self.time_fast * self.bandwidth
        self.demand_dpra[self.demand_dpra < 0] = 0

        self.individual_time_limit_dpra -= self.time_fast
        self.active_links_dpra[np.multiply(self.active_links_dpra, self.demand_dpra <= 0)] = 0 # 结束

        return UB_Rate, UU_Rate * 100

    def Compute_Rate(self, actions_power):


        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]

        # ------------ CM1 rate --------------------
        UB_Interference = np.zeros(self.n_RB)  # UB interference
        for i in range(len(self.users)):
            for j in range(self.n_neighbor):
                if not self.active_links_dpra[i, j]:
                    continue
                UB_Interference[actions[i][j]] += 10 ** ((self.UU_power_List[power_selection[i, j]] - self.UB_channels_fastfading[i, actions[i, j]]
                                                           + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Interference_dpra = UB_Interference + self.sig2
        UB_Signals = 10 ** ((self.UB_power - self.UB_channels_fastfading.diagonal() + self.userAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        UB_Rate = np.log2(1 + np.divide(UB_Signals, UB_Interference_dpra))

        # ------------ CM2 rate -------------------------
        UU_Interference = np.zeros((len(self.users), self.n_neighbor))
        UU_Signal = np.zeros((len(self.users), self.n_neighbor))
        actions[(np.logical_not(self.active_links_dpra))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.users[indexes[j, 0]].destinations[indexes[j, 1]]
                UU_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.UU_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

                UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i, receiver_j, i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)


                for k in range(j + 1, len(indexes)):
                    receiver_k = self.users[indexes[k][0]].destinations[indexes[k][1]]
                    UU_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
                    UU_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.UU_power_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.UU_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        UU_Interference_dpra = UU_Interference + self.sig2
        UU_Rate = np.log2(1 + np.divide(UU_Signal, UU_Interference_dpra))

        return UB_Rate, UU_Rate


    def Compute_Interference(self, actions):
        UU_Interference = np.zeros((len(self.users), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        for i in range(self.n_RB):
            for k in range(len(self.users)):
                for m in range(len(channel_selection[k, :])):
                    UU_Interference[k, m, i] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i][self.users[k].destinations[m]][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

        for i in range(len(self.users)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.users)):
                    for m in range(len(channel_selection[k, :])):
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        UU_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.UU_power_List[power_selection[i, j]]
                                                                                   - self.UU_channels_with_fastfading[i][self.users[k].destinations[m]][channel_selection[i, j]] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        self.UU_Interference_all = 10 * np.log10(UU_Interference)


    def Compute_Interference_sarl(self, actions):
        UU_Interference = np.zeros((len(self.users), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links_sarl)] = -1

        for i in range(self.n_RB):
            for k in range(len(self.users)):
                for m in range(len(channel_selection[k, :])):
                    UU_Interference[k, m, i] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i][self.users[k].destinations[m]][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

        for i in range(len(self.users)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.users)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        UU_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.UU_power_List[power_selection[i, j]]
                                                                                   - self.UU_channels_with_fastfading[i][self.users[k].destinations[m]][channel_selection[i, j]] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        self.UU_Interference_all_single = 10 * np.log10(UU_Interference)


    def Compute_Interference_dpra(self, actions):
        UU_Interference = np.zeros((len(self.users), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links_sarl)] = -1

        for i in range(self.n_RB):
            for k in range(len(self.users)):
                for m in range(len(channel_selection[k, :])):
                    UU_Interference[k, m, i] += 10 ** ((self.UB_power - self.UU_channels_with_fastfading[i][self.users[k].destinations[m]][i] + 2 * self.userAntGain - self.userNoiseFigure) / 10)

        for i in range(len(self.users)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.users)):
                    for m in range(len(channel_selection[k, :])):
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        UU_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.UU_power_List[power_selection[i, j]]
                                                                                   - self.UU_channels_with_fastfading[i][self.users[k].destinations[m]][channel_selection[i, j]] + 2 * self.userAntGain - self.userNoiseFigure) / 10)
        self.UU_Interference_all_dpra = 10 * np.log10(UU_Interference)

    def act_for_training(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)

        lambdda = 0
        reward = lambdda * np.sum(UB_Rate) / (self.n_User * 10) + (1 - lambdda) * np.sum(reward_elements) / (self.n_User * self.n_neighbor)

        return reward

    def act_for_testing(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
        UU_success = 1 - np.sum(self.active_links) / (self.n_User * self.n_neighbor)  # CM1 success rates

        return UB_Rate, UU_success, UU_Rate

    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
        UU_success = 1 - np.sum(self.active_links_rand) / (self.n_User * self.n_neighbor)  # CM2 success rates

        return UB_Rate, UU_success, UU_Rate

    def act_for_testing_sarl(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate = self.Compute_Performance_Reward_Test_sarl(action_temp)
        UU_success = 1 - np.sum(self.active_links_sarl) / (self.n_User * self.n_neighbor)  # CM2 rates

        return UB_Rate, UU_success, UU_Rate

    def act_for_testing_dpra(self, actions):

        action_temp = actions.copy()
        UB_Rate, UU_Rate = self.Compute_Performance_Reward_Test_dpra(action_temp)
        UU_success = 1 - np.sum(self.active_links_dpra) / (self.n_User * self.n_neighbor)  # CM2  rates

        return UB_Rate, UU_success, UU_Rate


    def new_random_game(self, n_User=0):

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


        self.demand_rand = self.demand_size * np.ones((self.n_User, self.n_neighbor))
        self.individual_time_limit_rand = self.time_slow * np.ones((self.n_User, self.n_neighbor))
        self.active_links_rand = np.ones((self.n_User, self.n_neighbor), dtype='bool')


        self.demand_sarl = self.demand_size * np.ones((self.n_User, self.n_neighbor))
        self.individual_time_limit_sarl = self.time_slow * np.ones((self.n_User, self.n_neighbor))
        self.active_links_sarl = np.ones((self.n_User, self.n_neighbor), dtype='bool')


        self.demand_dpra = self.demand_size * np.ones((self.n_User, self.n_neighbor))
        self.individual_time_limit_dpra = self.time_slow * np.ones((self.n_User, self.n_neighbor))
        self.active_links_dpra = np.ones((self.n_User, self.n_neighbor), dtype='bool')


