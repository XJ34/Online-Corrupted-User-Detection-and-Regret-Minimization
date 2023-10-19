from random import random

import numpy as np
from utlis import generate_items


def get_best_reward(items, theta):
    return np.max(np.dot(items, theta))


class Environment:
    # p: frequency vector of users
    def __init__(self, L, d, m, num_users, p, theta):
        self.L = L
        self.d = d
        self.p = p  # probability distribution over users
        self.rewards = [0]
        self.corruptions = [0]
        self.items = generate_items(num_items=1000, d=d)
        self.index_t = np.arange(1000)
        self.items_t = generate_items(20, 50)
        self.theta = theta

    def get_items(self):
        np.random.shuffle(self.index_t)
        self.items_t = self.items[self.index_t[0:20]]
        return self.items_t

    def feedback(self, i, k):
        x = self.items_t[k, :]
        r = np.dot(self.theta[i], x)
        
        if len(self.corruptions) < 20000:
            c = 2 * r
            self.corruptions.append(c)
            r_c = -r
        else:
            self.corruptions.append(0)
            r_c = r


        # if self.corruptions[-1] == 0:
        #     self.cp += 0.2
        # if random() < self.cp:
        #
        #     corruption = np.random.uniform(-0.2, 0.1, size=(1, ))
        #     r = r + corruption
        #     self.corruptions.append(corruption)
        #     self.cp = 0
        # else:
        #     self.corruptions.append(0)
        # print(self.corruptions)
        # r = np.dot(self.theta[i], x)
        # self.rewards.append(r)
        #
        # if r - 1.1 * self.rewards[-1] >= 0:
        #     corruption = np.random.uniform(-0.05, 0, size=(1, ))
        #
        #     r_c = np.dot(self.theta[i], x) + corruption
        #     y = r_c + np.random.normal(0, 0.1, size=1)
        # else:
        #     y = r + np.random.normal(0, 0.1, size=1)

        y = r_c + np.random.normal(0, 0.1, size=1)
        br = get_best_reward(self.items_t, self.theta[i])
        return y, r, br

    def generate_users(self):
        X = np.random.multinomial(1, self.p)
        I = np.nonzero(X)[0]
        return I
