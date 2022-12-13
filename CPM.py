import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class CPM:

    def __init__(self, data):
        self.data = data
        self.enum_data = {x: index for index, x in enumerate(data.loc[:, 'Task'])}
        self.relation_matrix = np.zeros((len(self.enum_data), len(self.enum_data)))
        self.graph_edge = []
        self.relation_list = {}
        self.enum_task = {'SP': 0}

        enum_data_l = list(self.enum_data.keys())
        for pt, t in zip(self.data.loc[:, 'Previous_task'], self.data.loc[:, 'Task']):
            if pd.isna(pt) is False:
                pt_separate = pt.split(',')
                for i in pt_separate:
                    self.relation_list[(i, t)] = self.data.loc[self.enum_data[i], 'Time']
                    self.graph_edge.append((i, t, self.data.loc[self.enum_data[i], 'Time']))
                    self.relation_matrix[self.enum_data[i], self.enum_data[t]] = 1
                    if i in enum_data_l:
                        enum_data_l.remove(i)
            else:
                self.relation_list['SP', t] = 0
                self.graph_edge.append(('SP', t, 0))

        for i in enum_data_l:
            self.relation_list[i, 'EP'] = self.data.loc[self.enum_data[i], 'Time']
            self.graph_edge.append((i, 'EP', self.data.loc[self.enum_data[i], 'Time']))

        index = 1
        while not np.all(self.relation_matrix == -1):
            for i, col in enumerate(self.relation_matrix.copy().T):
                if np.all(np.logical_or(col == 0, col == -1)) and np.any(col == 0):
                    self.relation_matrix[i, :] = [-1] * len(self.relation_matrix)
                    self.relation_matrix[:, i] = [-1] * len(self.relation_matrix)
                    self.enum_task[data.loc[i, 'Task']] = index
                    index += 1
        self.enum_task['EP'] = index

    def define_deadlines(self):
        Tw: List[Optional[int]] = [None] * len(self.enum_task)  # najpóźniejszy termin
        Tp: List[Optional[int]] = [None] * len(self.enum_task)  # najwcześniejszy termin
        z: List[Optional[int]] = [None] * len(self.enum_task)  # zapas

        for i in self.enum_task.keys():
            maximum = 0
            for j in self.enum_task.keys():
                if (j, i) in self.relation_list:
                    if Tw[self.enum_task[j]] + self.relation_list[(j, i)] > maximum:
                        maximum = Tw[self.enum_task[j]] + self.relation_list[(j, i)]
            Tw[self.enum_task[i]] = maximum

        for i in reversed(self.enum_task.keys()):
            minimum = Tw[-1]
            for j in reversed(self.enum_task.keys()):
                if (i, j) in self.relation_list:
                    if Tp[self.enum_task[j]] - self.relation_list[(i, j)] < minimum:
                        minimum = Tp[self.enum_task[j]] - self.relation_list[(i, j)]
            Tp[self.enum_task[i]] = minimum
            z[self.enum_task[i]] = Tp[self.enum_task[i]] - Tw[self.enum_task[i]]

        return Tw, Tp, z

    def get_critical_path(self, z):
        path = ['SP']
        idx = 1
        for e, i in zip(z, self.enum_task):
            if e == 0 and (path[idx - 1], i) in self.relation_list:
                path.append(i)
                idx += 1

        return path

# wyświeltanie gafu
def Graph_CPM(edge):
    G = nx.Graph()
    G.add_weighted_edges_from(edge)
    options = {
        "font_size": 16,
        "node_size": 1000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 3,
        "width": 3,
        "with_labels": True
    }
    nx.draw(G, **options)
    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, edge_labels)
    plt.show()

