import os
import pickle as pkl
from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.sparse as sp

dataf = "raw_data/"
data_outputf = "data/"

CITESEER_LABEL_MAPPING = {"DB": 0, "IR": 1, "Agents": 2, "ML": 3, "HCI": 4, "AI": 5}
CORA_LABEL_MAPPING = {
    "Case_Based": 0,
    "Genetic_Algorithms": 1,
    "Neural_Networks": 2,
    "Probabilistic_Methods": 3,
    "Reinforcement_Learning": 4,
    "Rule_Learning": 5,
    "Theory": 6,
}


def generate_citation_dataset(name, label_mapping):
    """
    Generates the following files:
    _ ind.cora.allx: <1708x1433 sparse matrix of type '<class 'numpy.float32'>'
        with 31261 stored elements in Compressed Sparse Row format>
    - ind.cora.ally: numpy.ndarray int32 of shape (1708, 7)  (one hot encoding)
    - ind.cora.graph: defaultdict(list)
    """

    print(f"[+] Preprocessing {name} dataset")
    # load raw data
    cites = open(dataf + f"{name}.cites").readlines()
    content = open(dataf + f"{name}.content").readlines()

    # generate node id mappings
    content = [line.strip().split("\t") for line in content]
    node_list = [r[0] for r in content]
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # convert content to int
    width = len(content[0])
    for r in content:
        # ignore first and last, node id and label
        for i in range(1, width - 1):
            r[i] = int(r[i])
        r[-1] = label_mapping[r[-1]]
        r[0] = node_to_idx[r[0]]

    X = np.array(content)
    num_nodes = len(X)
    # ignore the first column as it is the node id
    allx, y = X[:, 1 : width - 1], X[:, -1]
    allx = allx.astype(float)

    # convert y to one hot encoding
    num_classes = len(label_mapping)
    ally = np.zeros((num_nodes, num_classes))
    ally[np.arange(num_nodes), y] = 1

    # create label index file
    y_index = [[] for _ in range(num_classes)]
    for idx, y_val in enumerate(y):
        y_index[y_val].append(idx)

    # create networkx Graph and convert to adj
    cites = [list(l.strip().split("\t")) for l in cites]
    graph = defaultdict(list)
    edges, skipped = 0, 0
    for cited, citing in cites:
        if cited not in node_to_idx:
            # print(f'paper {cited} not in content files, skippinng creating edge')
            skipped += 1
            continue
        if citing not in node_to_idx:
            # print(f'paper {citing} not in content files, skippinng creating edge')
            skipped += 1
            continue
        edges += 1
        graph[node_to_idx[citing]].append(node_to_idx[cited])

    # save the following files: ally, allx, directed_graph
    if not os.path.exists(data_outputf):
        os.makedirs(data_outputf)

    # print summary stats
    print(f"\t- {num_nodes} nodes")
    print(f"\t- {edges} edges, skipped {skipped} edges")
    print(f"\t- {num_classes} labels")

    pkl.dump(ally, open(f"{data_outputf}ind.{name}.ally", "wb"))
    pkl.dump(allx, open(f"{data_outputf}ind.{name}.allx", "wb"))
    pkl.dump(graph, open(f"{data_outputf}ind.{name}.graph", "wb"))
    pkl.dump(y_index, open(f"{data_outputf}ind.{name}.y.index", "wb"))


def generate_pubmed_dataset():
    print(f"[+] Preprocessing pubmed dataset")

    items = ["Pubmed-Diabetes.DIRECTED.cites.tab", "Pubmed-Diabetes.NODE.paper.tab"]
    cites, content = [open(dataf + item).readlines() for item in items]

    column_data = content[1]
    columns = column_data.split("\t")[1:-1]
    columns = [c.split(":")[1] for c in columns]  # {type}:{name}:0.0
    assert len(columns) == 500, "expecting 500 features"

    feature_to_idx = {c: idx for idx, c in enumerate(columns)}

    content = content[2:]  # remove the first 2 rows
    assert len(content) == 19717, "expect 19717 nodes"

    num_nodes = 19717
    content = [row.split("\t") for row in content]

    allx = np.zeros((num_nodes, 500))
    for idx, row in enumerate(content):
        # {id},label={1.2.3},{feature_name}={val},...,summary={list_features}
        for node_features in row[2:-1]:
            feat, val = node_features.split("=")
            allx[idx, feature_to_idx[feat]] = float(val)

    # convert y to one hot encoding
    num_classes = 3
    y = [int(row[1].split("=")[1]) - 1 for row in content]
    ally = np.zeros((num_nodes, num_classes))
    ally[np.arange(num_nodes), y] = 1

    # create label index file
    y_index = [[] for _ in range(num_classes)]
    for idx, y_val in enumerate(y):
        y_index[y_val].append(idx)

    node_ids = [int(row[0]) for row in content]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    def get_citing_cited(row):
        # A cites B
        # '37511\tpaper:19668377\t|\tpaper:17293876\n'
        row = row.split("\t")
        citing, cited = row[1], row[3]
        citing = int(citing.split(":")[1])
        cited = int(cited.split(":")[1])
        return citing, cited

    edges = [get_citing_cited(r) for r in cites[2:]]
    assert len(edges) == 44338
    num_edges, skipped = 0, 0
    graph = defaultdict(list)
    for citing, cited in edges:
        if cited not in node_to_idx:
            skipped += 1
            continue
        if citing not in node_to_idx:
            skipped += 1
            continue
        num_edges += 1
        graph[node_to_idx[citing]].append(node_to_idx[cited])

    # save the following files: ally, allx, directed_graph
    if not os.path.exists(data_outputf):
        os.makedirs(data_outputf)

    # print summary stats
    print(f"\t- {num_nodes} nodes")
    print(f"\t- {num_edges} edges, skipped {skipped} edges")
    print(f"\t- {num_classes} labels")

    pkl.dump(ally, open(f"{data_outputf}ind.pubmed.ally", "wb"))
    pkl.dump(allx, open(f"{data_outputf}ind.pubmed.allx", "wb"))
    pkl.dump(graph, open(f"{data_outputf}ind.pubmed.graph", "wb"))
    pkl.dump(y_index, open(f"{data_outputf}ind.pubmed.y.index", "wb"))


if __name__ == "__main__":
    generate_citation_dataset("cora", CORA_LABEL_MAPPING)
    generate_citation_dataset("citeseer", CITESEER_LABEL_MAPPING)
    generate_pubmed_dataset()
