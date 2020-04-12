import os.path as osp
import pickle
import sys
from itertools import repeat

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


def read_citation_data(folder, prefix, directed):
    names = ["allx", "ally", "graph", "y.index"]
    items = [read_file(folder, prefix, name) for name in names]
    allx, ally, graph, y_index = items

    # - 80% for training
    # - 10% for validation
    # - 10% for testing
    N = len(ally)
    index = np.arange(0, N)
    np.random.shuffle(index)
    num_train = int(N * 0.8)
    num_val = int(N * 0.1)
    idx_train = index[:num_train]
    idx_val = index[num_train : num_train + num_val]
    idx_test = index[num_train + num_val :]

    ally = ally.max(dim=1)[1]

    train_mask = index_to_mask(idx_train, size=ally.size(0))
    val_mask = index_to_mask(idx_val, size=ally.size(0))
    test_mask = index_to_mask(idx_test, size=ally.size(0))

    if not directed:
        edges_to_add = []
        for node, nbrs in graph.items():
            for nbr in nbrs:
                edges_to_add.append((nbr, node))
        for u, v in edges_to_add:
            graph[u].append(v)

    edge_index = edge_index_from_dict(graph, num_nodes=ally.size(0))

    data = Data(x=allx, edge_index=edge_index, y=ally)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, "ind.{}.{}".format(prefix.lower(), name))

    if name == "test.index":
        return read_txt_array(path, dtype=torch.long)

    with open(path, "rb") as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding="latin1")
        else:
            out = pickle.load(f)

    if name == "graph" or name == "y.index":
        return out

    out = out.todense() if hasattr(out, "todense") else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def get_directed_dataset(name, directed, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "directed", name)
    dataset = DirectedCitation(path, name, directed)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "planetoid", name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


class DirectedCitation(InMemoryDataset):
    """
    Args:
    root (string): Root directory where the dataset should be saved.
    name (string): The name of the dataset (:obj:`"Cora"`,
        :obj:`"CiteSeer"`, :obj:`"PubMed"`).
    direted (bool): whether want directed (default: True) 
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/raynoldng/directed-citation/raw/master/data'

    def __init__(self, root, name, directed=True, transform=None, pre_transform=None):
        """not overriding the download() and process() methods
        """
        self.name = name
        self.directed = directed
        super(DirectedCitation, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ["allx", "ally", "graph", "y.index"]
        return ["ind.{}.{}".format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return "{}.{}.pt".format(
            self.name, "directed" if self.directed else "undirected"
        )

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_citation_data(self.raw_dir, self.name, self.directed)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)
