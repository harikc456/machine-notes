"""Tree dataset generation for hyperbolic validation experiment."""

import numpy as np
import networkx as nx
import torch
from typing import Tuple, Dict, List
import pickle
import os


class TreeDataset:
    """Generate and load synthetic tree data."""

    def __init__(self, num_nodes: int = 1000, branching_factor: int = 4,
                 depth: int = 5, embedding_dim: int = 64, seed: int = 42):
        """
        Initialize tree dataset parameters.

        Args:
            num_nodes: Total number of nodes in the tree
            branching_factor: Number of children per node
            depth: Maximum depth of the tree
            embedding_dim: Dimension of node features
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.branching_factor = branching_factor
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.seed = seed

        self.tree = None
        self.node_features = None
        self.node_depths = None
        self.edges = None
        self.non_edges = None
        self.node_to_depth = {}
        self.node_to_parent = {}

    def generate_balanced_tree(self) -> nx.DiGraph:
        """Generate a balanced tree with random walk features."""
        np.random.seed(self.seed)

        # Create a balanced tree using NetworkX
        # For a balanced tree with branching factor r and height h,
        # total nodes = (r^(h+1) - 1) / (r - 1)
        # We'll use this formula to determine actual tree size

        self.tree = nx.DiGraph()
        self.tree.add_node(0)
        self.node_to_depth[0] = 0
        self.node_to_parent[0] = None

        current_node = 1
        queue = [0]

        # Build balanced tree
        while queue and current_node < self.num_nodes:
            parent = queue.pop(0)
            parent_depth = self.node_to_depth[parent]

            if parent_depth < self.depth:
                # Add children
                for _ in range(self.branching_factor):
                    if current_node >= self.num_nodes:
                        break
                    self.tree.add_edge(parent, current_node)
                    self.node_to_depth[current_node] = parent_depth + 1
                    self.node_to_parent[current_node] = parent
                    queue.append(current_node)
                    current_node += 1

        self.num_nodes = current_node  # Actual number of nodes
        self.node_depths = np.array([self.node_to_depth[i] for i in range(self.num_nodes)])

        # Generate features using random walk on tree
        self._generate_features()

        # Generate edges and non-edges
        self._generate_edge_sets()

        return self.tree

    def _generate_features(self):
        """Generate node features using random walk with Gaussian noise."""
        # Initialize features randomly
        features = np.random.randn(self.num_nodes, self.embedding_dim) * 0.1

        # Perform random walk to propagate features along tree structure
        num_walks = 10
        walk_length = 5

        for start_node in range(self.num_nodes):
            for _ in range(num_walks):
                current = start_node
                for step in range(walk_length):
                    # Get neighbors (parent and children)
                    neighbors = list(self.tree.predecessors(current)) + \
                               list(self.tree.successors(current))
                    if not neighbors:
                        break
                    # Move to random neighbor
                    current = np.random.choice(neighbors)
                    # Add contribution to features
                    features[start_node] += np.random.randn(self.embedding_dim) * 0.1

        # Add Gaussian noise
        features += np.random.randn(self.num_nodes, self.embedding_dim) * 0.1

        # Normalize features
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        self.node_features = torch.FloatTensor(features)

    def _generate_edge_sets(self):
        """Generate edge and non-edge sets for link prediction with hard negatives."""
        # Get all edges (parent-child relationships)
        self.edges = list(self.tree.edges())

        # Build parent-to-children mapping for sibling identification
        parent_to_children = {}
        for u, v in self.edges:
            if u not in parent_to_children:
                parent_to_children[u] = []
            parent_to_children[u].append(v)

        # Generate non-edges with different difficulty levels
        # Type 1: Siblings (same parent) - HARD negatives, tree distance = 2
        sibling_pairs = []
        for parent, children in parent_to_children.items():
            if len(children) >= 2:
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        u, v = children[i], children[j]
                        if u < v:
                            sibling_pairs.append((u, v))
                        else:
                            sibling_pairs.append((v, u))

        # Type 2: Distant nodes (different subtrees) - EASY negatives
        edge_set = set(self.edges)
        sibling_set = set(sibling_pairs)
        all_pairs = set()
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                all_pairs.add((i, j))

        distant_pairs = list(all_pairs - edge_set - sibling_set)

        # Sample non-edges: 50% siblings (hard), 50% distant (easy)
        num_siblings = min(len(sibling_pairs), len(self.edges) // 2)
        num_distant = len(self.edges) - num_siblings

        sampled_siblings = np.random.choice(
            len(sibling_pairs), size=num_siblings, replace=False
        ) if sibling_pairs else []
        sampled_distant = np.random.choice(
            len(distant_pairs), size=min(num_distant, len(distant_pairs)), replace=False
        ) if distant_pairs else []

        self.non_edges = [sibling_pairs[i] for i in sampled_siblings] + \
                         [distant_pairs[i] for i in sampled_distant]
        np.random.shuffle(self.non_edges)  # Mix hard and easy negatives

        # Store sibling pairs for reference
        self.sibling_pairs = sibling_pairs

    def get_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict:
        """
        Split edges into train/val/test sets.

        Args:
            train_ratio: Proportion of edges for training
            val_ratio: Proportion of edges for validation (rest goes to test)

        Returns:
            Dictionary with train/val/test edges and non-edges
        """
        if self.edges is None:
            raise ValueError("Must call generate_balanced_tree first")

        np.random.seed(self.seed)

        # Shuffle edges
        edges_shuffled = self.edges.copy()
        np.random.shuffle(edges_shuffled)

        # Split edges
        n_train = int(len(edges_shuffled) * train_ratio)
        n_val = int(len(edges_shuffled) * val_ratio)

        train_edges = edges_shuffled[:n_train]
        val_edges = edges_shuffled[n_train:n_train + n_val]
        test_edges = edges_shuffled[n_train + n_val:]

        # Split non-edges similarly
        non_edges_shuffled = self.non_edges.copy()
        np.random.shuffle(non_edges_shuffled)

        n_train_neg = int(len(non_edges_shuffled) * train_ratio)
        n_val_neg = int(len(non_edges_shuffled) * val_ratio)

        train_non_edges = non_edges_shuffled[:n_train_neg]
        val_non_edges = non_edges_shuffled[n_train_neg:n_train_neg + n_val_neg]
        test_non_edges = non_edges_shuffled[n_train_neg + n_val_neg:]

        return {
            'train_edges': torch.LongTensor(train_edges),
            'val_edges': torch.LongTensor(val_edges),
            'test_edges': torch.LongTensor(test_edges),
            'train_non_edges': torch.LongTensor(train_non_edges),
            'val_non_edges': torch.LongTensor(val_non_edges),
            'test_non_edges': torch.LongTensor(test_non_edges),
            'num_nodes': self.num_nodes,
            'node_features': self.node_features,
            'node_depths': self.node_depths
        }

    def get_tree_distance(self, u: int, v: int) -> int:
        """Compute tree distance between two nodes (shortest path)."""
        try:
            # Convert directed to undirected for path finding
            undirected_tree = self.tree.to_undirected()
            return nx.shortest_path_length(undirected_tree, u, v)
        except nx.NetworkXNoPath:
            return -1  # No path (shouldn't happen in a tree)

    def save(self, path: str):
        """Save dataset to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'tree': self.tree,
                'node_features': self.node_features,
                'node_depths': self.node_depths,
                'edges': self.edges,
                'non_edges': self.non_edges,
                'node_to_depth': self.node_to_depth,
                'node_to_parent': self.node_to_parent,
                'num_nodes': self.num_nodes,
                'branching_factor': self.branching_factor,
                'depth': self.depth,
                'embedding_dim': self.embedding_dim
            }, f)

    @classmethod
    def load(cls, path: str) -> 'TreeDataset':
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        dataset = cls(
            num_nodes=data['num_nodes'],
            branching_factor=data['branching_factor'],
            depth=data['depth'],
            embedding_dim=data['embedding_dim']
        )
        dataset.tree = data['tree']
        dataset.node_features = data['node_features']
        dataset.node_depths = data['node_depths']
        dataset.edges = data['edges']
        dataset.non_edges = data['non_edges']
        dataset.node_to_depth = data['node_to_depth']
        dataset.node_to_parent = data['node_to_parent']

        return dataset


def generate_dataset(save_dir: str = 'experiments/tree_validation/data',
                     num_nodes: int = 1000,
                     branching_factor: int = 4,
                     depth: int = 5,
                     embedding_dim: int = 64,
                     seed: int = 42) -> TreeDataset:
    """
    Generate and save a tree dataset.

    Args:
        save_dir: Directory to save the dataset
        num_nodes: Number of nodes
        branching_factor: Branching factor
        depth: Tree depth
        embedding_dim: Feature dimension
        seed: Random seed

    Returns:
        TreeDataset instance
    """
    print(f"Generating tree dataset with {num_nodes} nodes...")
    dataset = TreeDataset(
        num_nodes=num_nodes,
        branching_factor=branching_factor,
        depth=depth,
        embedding_dim=embedding_dim,
        seed=seed
    )
    dataset.generate_balanced_tree()

    save_path = os.path.join(save_dir, 'tree_data.pkl')
    dataset.save(save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Actual nodes: {dataset.num_nodes}")
    print(f"Edges: {len(dataset.edges)}")
    print(f"Non-edges: {len(dataset.non_edges)}")

    return dataset


if __name__ == '__main__':
    # Test dataset generation
    dataset = generate_dataset()
    splits = dataset.get_splits()
    print(f"\nDataset splits:")
    print(f"  Train edges: {len(splits['train_edges'])}")
    print(f"  Val edges: {len(splits['val_edges'])}")
    print(f"  Test edges: {len(splits['test_edges'])}")
    print(f"  Feature shape: {splits['node_features'].shape}")
