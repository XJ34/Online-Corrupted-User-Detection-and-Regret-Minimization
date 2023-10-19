import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

from utlis import edge_probability, is_power2, isInvertible



class Cluster:
    def __init__(self, users, S, b, N):
        self.users = users  # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)


class CLUB_co():
    # random_init: use random initialization or not
    def __init__(self, nu, d, T, edge_probability=1):

        self.nu = nu
        self.d = d
        self.T = T
        # self.alpha = 4 * np.sqrt(d) # parameter for cut edge
        self.G = nx.gnp_random_graph(nu, edge_probability)
        self.clusters = {0: Cluster(users=range(nu), S=np.eye(d), b=np.zeros(d), N=0)}
        self.cluster_inds = np.zeros(nu)
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.S_true = {i: np.eye(d) for i in range(nu)}
        self.b_true = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}
        self.w = {i: 1 for i in range(nu)}
        self.rewards = np.zeros(self.T)
        self.best_rewards = np.zeros(self.T)

        self.N = np.zeros(nu)

        self.num_clusters = np.zeros(T)

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        return np.argmax(np.dot(items, cluster.theta) + 1.5 * (np.matmul(items, cluster.Sinv) * items).sum(axis=1))


    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br

        self.S[i] += self.w[i] * np.outer(x, x)
        self.b[i] += self.w[i] * y * x
        self.S_true[i] += np.outer(x, x)
        self.b_true[i] += y * x
        self.N[i] += 1

        self.Sinv[i] = np.linalg.inv(self.S[i])
        self.theta[i] = np.matmul(self.Sinv[i], self.b[i])
        #x = np.reshape(x, (50, 1))

        self.w[i] = min(1, float(0.5 / np.sqrt(np.dot(np.matmul(x.T, self.Sinv[i]), x))))

        c = self.cluster_inds[i]
        self.clusters[c].S += self.w[i] * np.outer(x, x)
        self.clusters[c].b += self.w[i] * y * x
        self.clusters[c].N += 1

        self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
        self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)
    def _if_split(self, theta, N1, N2):
        # alpha = 2 * np.sqrt(2 * self.d)
        alpha = 1
        alpha2 = 1

        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))

        return np.linalg.norm(theta) > alpha * (_factT(N1) + _factT(N2) + alpha2)

    def update(self, t):
        update_clusters = False
        for i in self.I:
            c = self.cluster_inds[i]

            A = [a for a in self.G.neighbors(i)]
            for j in A:
                if self.N[i] and self.N[j] and self._if_split(self.theta[i] - self.theta[j], self.N[i], self.N[j]):
                    self.G.remove_edge(i, j)

                    update_clusters = True

        if update_clusters:
            C = set()
            for i in self.I:  # suppose there is only one user per round
                C = nx.node_connected_component(self.G, i)
                if len(C) < len(self.clusters[c].users):
                    remain_users = set(self.clusters[c].users)
                    self.clusters[c] = Cluster(list(C), S=sum([self.S[k] - np.eye(self.d) for k in C]) + np.eye(self.d),
                                               b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))

                    remain_users = remain_users - set(C)
                    c = max(self.clusters) + 1
                    while len(remain_users) > 0:
                        j = np.random.choice(list(remain_users))
                        C = nx.node_connected_component(self.G, j)

                        self.clusters[c] = Cluster(list(C),
                                                   S=sum([self.S[k] - np.eye(self.d) for k in C]) + np.eye(self.d),
                                                   b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))
                        for j in C:
                            self.cluster_inds[j] = c

                        c += 1
                        remain_users = remain_users - set(C)

            # print(len(self.clusters))

        self.num_clusters[t] = len(self.clusters)

    def run(self, envir):
        for t in range(self.T):
            if t % 10000 == 0:
                cluster_list = []
                for i in self.cluster_inds:
                    address_index = [x for x in range(len(self.cluster_inds)) if self.cluster_inds[x] == i]
                    cluster_list.append([i, address_index])
                dict_address = dict(cluster_list)
                bot_index = []
                for i in dict_address:
                    S_i = np.eye(50)
                    b_i = np.zeros(50)
                    for j in dict_address[i]:
                        S_i += self.S[j] - np.eye(50)
                        b_i += self.b[j]
                    thetai = np.matmul(np.linalg.inv(S_i), b_i)
                    for j in dict_address[i]:
                        thetaj = np.matmul(np.linalg.inv(self.S_true[j]), self.b_true[j])

                        # print(np.linalg.norm(thetaj - m[j]))
                        def _factT(T):
                            return np.sqrt((1 + np.log(1 + T)) / (1 + T))

                        if np.linalg.norm(thetaj - thetai) > 8 * (_factT(self.T/self.nu) + _factT(len(dict_address[i]) * self.T/self.nu)):
                            bot_index.append(j)

            self.I = envir.generate_users()

            for i in self.I:
                items = envir.get_items()
                kk = self.recommend(i=i, items=items, t=t)
                x = items[kk]
                y, r, br = envir.feedback(i=i, k=kk)
                self.store_info(i=i, x=x, y=y, t=t, r=r, br=br)

            self.update(t)

        print()
