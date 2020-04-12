import argparse
import os.path as osp
import pickle
from collections import defaultdict

import numpy as np


def read_file(folder, prefix, name):
    path = osp.join(folder, "ind.{}.{}".format(prefix.lower(), name))

    with open(path, "rb") as f:
        out = pickle.load(f, encoding="latin1")
    return out


def run(dataset):
    items = ["graph", "ally"]
    graph, ally = [read_file("data", dataset, item) for item in items]
    ally = np.argmax(ally, axis=1) # undo one hot encoding
    undirected_graph = defaultdict(list)
    reverse_graph = defaultdict(list)

    for u, nbrs in graph.items():
        for v in nbrs:
            reverse_graph[v].append(u)
            undirected_graph[u].append(v)
            undirected_graph[v].append(u)

    f1, f2 = homophility(graph, ally)
    b1, b2 = homophility(reverse_graph, ally)
    s1, s2 = homophility(undirected_graph, ally)

    print("Forward graph: {:.3f} ± {:.3f}".format(f1, f2))
    print("Reverse graph: {:.3f} ± {:.3f}".format(b1, b2))
    print("Undirected graph: {:.3f} ± {:.3f}".format(s1, s2))


def homophility(graph, ally):
    scores = []
    for x in range(len(ally)):
        y = ally[x]
        if graph[x]:
            same_label = len([nbr for nbr in graph[x] if ally[nbr] == y])
            scores.append(same_label / len(graph[x]))
        else:
            scores.append(0)
    return np.mean(scores), np.std(scores)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

run(args.dataset)