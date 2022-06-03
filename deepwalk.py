import networkx as nx
import random
from skip_gram import SkipGram
from typing import List
import numpy as np


class DeepWalk:
    """ https://github.com/benedekrozemberczki/karateclub """
    def __init__(
        self,
        walk_number: int,
        walk_length: int,
        n_emb_dim: int,  # a dimension of embedding weight
        window_size: int,
        epochs: int,
        learning_rate: float,
    ):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.n_emb_dim = n_emb_dim
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, graph: nx.classes.graph.Graph) -> List[float]:
        num_of_nodes = graph.number_of_nodes()

        self.model = SkipGram(
            n_input_dim=num_of_nodes,
            alpha=self.learning_rate,
            epochs=self.epochs,
            n_emb_dim=self.n_emb_dim,
            window_size=self.window_size,
        )

        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)

        # train a skip-gram model, get a list including losses and return the list
        return self.model.train(self.epochs, walker.walks, num_of_nodes, self.window_size)

    def embedding(self) -> np.array:
        return self.model.hidden_weight


class RandomWalker:
    def __init__(self, walk_length: int, walk_number: int):
        self.walk_length = walk_length
        self.walk_number = walk_number  # mini-batch size
        self.walks = None
        self.graph = None

    def do_walk(self, node) -> List[str]:
        walk = [node]
        for _ in range(self.walk_length - 1):
            neighbors = [node for node in self.graph.neighbors(walk[-1])]
            if len(neighbors) > 0:
                walk = walk + random.sample(neighbors, 1)  # sample a walk sequence randomly
        return [str(w) for w in walk]

    def do_walks(self, graph: nx.Graph) -> None:
        self.walks = []
        self.graph = graph
        for node in self.graph.nodes():
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
