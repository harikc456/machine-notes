"""Evaluation metrics for hyperbolic embedding experiments."""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os


def compute_auroc(model: torch.nn.Module,
                  edges: torch.Tensor,
                  non_edges: torch.Tensor,
                  batch_size: int = 256,
                  device: str = 'cpu') -> float:
    """
    Compute AUROC for link prediction.

    Args:
        model: Embedding model with distance method
        edges: Positive edge pairs
        non_edges: Negative edge pairs
        batch_size: Batch size for computation
        device: Device to use

    Returns:
        AUROC score
    """
    from sklearn.metrics import roc_auc_score

    model.eval()

    # Compute distances for positive edges
    pos_dists = []
    with torch.no_grad():
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size].to(device)
            dists = model.distance(batch[:, 0], batch[:, 1])
            pos_dists.append(dists.cpu())
    pos_dists = torch.cat(pos_dists).numpy()

    # Compute distances for negative edges
    neg_dists = []
    with torch.no_grad():
        for i in range(0, len(non_edges), batch_size):
            batch = non_edges[i:i+batch_size].to(device)
            dists = model.distance(batch[:, 0], batch[:, 1])
            neg_dists.append(dists.cpu())
    neg_dists = torch.cat(neg_dists).numpy()

    # Create labels and scores
    labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    scores = np.concatenate([-pos_dists, -neg_dists])  # Lower distance = higher score

    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.5

    return float(auroc)


def compute_tree_distance_correlation(model: torch.nn.Module,
                                      dataset,
                                      num_pairs: int = 1000,
                                      batch_size: int = 256,
                                      device: str = 'cpu') -> Dict[str, float]:
    """
    Compute correlation between learned distances and tree distances.

    Args:
        model: Embedding model
        dataset: TreeDataset instance with ground truth
        num_pairs: Number of node pairs to sample
        batch_size: Batch size for distance computation
        device: Device to use

    Returns:
        Dictionary with correlation metrics
    """
    model.eval()

    # Sample random node pairs
    np.random.seed(42)
    pairs = []
    tree_dists = []

    for _ in range(num_pairs):
        u = np.random.randint(0, dataset.num_nodes)
        v = np.random.randint(0, dataset.num_nodes)
        if u != v:
            tree_dist = dataset.get_tree_distance(u, v)
            if tree_dist >= 0:
                pairs.append((u, v))
                tree_dists.append(tree_dist)

    pairs = torch.LongTensor(pairs)
    tree_dists = np.array(tree_dists)

    # Compute learned distances
    learned_dists = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size].to(device)
            dists = model.distance(batch[:, 0], batch[:, 1])
            learned_dists.append(dists.cpu())
    learned_dists = torch.cat(learned_dists).numpy()

    # Compute correlations
    if len(tree_dists) > 10:
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(learned_dists, tree_dists)

        # Pearson correlation
        pearson_corr = np.corrcoef(learned_dists, tree_dists)[0, 1]

        return {
            'spearman_rho': float(spearman_corr),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_corr),
            'num_pairs': len(tree_dists)
        }
    else:
        return {
            'spearman_rho': 0.0,
            'spearman_p': 1.0,
            'pearson_r': 0.0,
            'num_pairs': len(tree_dists)
        }


def compute_norm_depth_correlation(model: torch.nn.Module,
                                   dataset,
                                   device: str = 'cpu') -> Dict[str, float]:
    """
    Compute correlation between embedding norm and tree depth.
    For hyperbolic embeddings, should show monotonic increase.

    Args:
        model: Embedding model
        dataset: TreeDataset with node_depths
        device: Device to use

    Returns:
        Dictionary with correlation metrics
    """
    model.eval()

    with torch.no_grad():
        # Get all embeddings
        all_indices = torch.arange(dataset.num_nodes, device=device)
        if hasattr(model, 'forward'):
            embeddings = model(all_indices)
        else:
            embeddings = model.embedding(all_indices)

        # Compute norms
        norms = torch.norm(embeddings, dim=-1).cpu().numpy()
        depths = dataset.node_depths

    # Compute correlations
    if len(depths) > 10:
        spearman_corr, spearman_p = spearmanr(norms, depths)
        pearson_corr = np.corrcoef(norms, depths)[0, 1]

        return {
            'spearman_rho': float(spearman_corr),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_corr),
            'mean_norm_by_depth': {
                int(d): float(np.mean(norms[depths == d]))
                for d in np.unique(depths)
            }
        }
    else:
        return {
            'spearman_rho': 0.0,
            'spearman_p': 1.0,
            'pearson_r': 0.0,
            'mean_norm_by_depth': {}
        }


def visualize_embeddings(model: torch.nn.Module,
                         dataset,
                         save_path: str,
                         device: str = 'cpu',
                         method: str = 'pca'):
    """
    Visualize embeddings colored by depth.

    Args:
        model: Embedding model
        dataset: TreeDataset
        save_path: Path to save figure
        device: Device to use
        method: Dimensionality reduction method ('pca' or 'umap')
    """
    model.eval()

    with torch.no_grad():
        # Get all embeddings
        all_indices = torch.arange(dataset.num_nodes)
        if hasattr(model, 'forward'):
            embeddings = model(all_indices)
        else:
            embeddings = model.embedding(all_indices)
        embeddings = embeddings.cpu().numpy()

    depths = dataset.node_depths

    # Reduce to 2D
    if embeddings.shape[1] == 2:
        emb_2d = embeddings
    else:
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            emb_2d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                emb_2d = reducer.fit_transform(embeddings)
            except ImportError:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                emb_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: 2D embedding space colored by depth
    ax1 = axes[0]
    scatter = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], c=depths,
                          cmap='viridis', alpha=0.6, s=20)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_title('Embeddings Colored by Tree Depth')
    plt.colorbar(scatter, ax=ax1, label='Tree Depth')

    # Plot 2: Norm vs Depth
    ax2 = axes[1]
    norms = np.linalg.norm(embeddings, axis=1)
    ax2.scatter(depths, norms, alpha=0.5, s=20)
    ax2.set_xlabel('Tree Depth')
    ax2.set_ylabel('Embedding Norm')
    ax2.set_title('Embedding Norm vs Tree Depth')

    # Add trend line
    unique_depths = np.unique(depths)
    mean_norms = [np.mean(norms[depths == d]) for d in unique_depths]
    ax2.plot(unique_depths, mean_norms, 'r-', linewidth=2, label='Mean norm')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model: torch.nn.Module,
                   dataset,
                   split_data: Dict,
                   batch_size: int = 256,
                   device: str = 'cpu') -> Dict:
    """
    Run full evaluation suite on a model.

    Args:
        model: Trained embedding model
        dataset: TreeDataset instance
        split_data: Dictionary with train/val/test splits
        batch_size: Batch size for evaluation
        device: Device to use

    Returns:
        Dictionary with all evaluation metrics
    """
    print("Computing metrics...")

    # 1. Link prediction AUROC
    print("  - Computing AUROC...")
    val_auroc = compute_auroc(
        model,
        split_data['val_edges'],
        split_data['val_non_edges'],
        batch_size=batch_size,
        device=device
    )

    test_auroc = compute_auroc(
        model,
        split_data['test_edges'],
        split_data['test_non_edges'],
        batch_size=batch_size,
        device=device
    )

    # 2. Tree distance correlation
    print("  - Computing tree distance correlation...")
    tree_corr = compute_tree_distance_correlation(
        model, dataset,
        num_pairs=min(2000, dataset.num_nodes * 5),
        batch_size=batch_size,
        device=device
    )

    # 3. Norm vs depth correlation
    print("  - Computing norm-depth correlation...")
    norm_depth = compute_norm_depth_correlation(model, dataset, device=device)

    results = {
        'val_auroc': val_auroc,
        'test_auroc': test_auroc,
        'tree_distance_spearman': tree_corr['spearman_rho'],
        'tree_distance_pearson': tree_corr['pearson_r'],
        'norm_depth_spearman': norm_depth['spearman_rho'],
        'norm_depth_pearson': norm_depth['pearson_r'],
        'tree_distance_corr_details': tree_corr,
        'norm_depth_corr_details': norm_depth
    }

    return results


def print_results(results: Dict, model_name: str = "Model"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Validation AUROC:     {results['val_auroc']:.4f}")
    print(f"Test AUROC:           {results['test_auroc']:.4f}")
    print(f"Tree Dist Correlation (Spearman): {results['tree_distance_spearman']:.4f}")
    print(f"Norm-Depth Correlation (Spearman): {results['norm_depth_spearman']:.4f}")
    print(f"{'='*60}")


def compare_results(euclidean_results: Dict,
                   hyperbolic_results: Dict) -> Dict:
    """
    Compare Euclidean and Hyperbolic results.

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'val_auroc_diff': hyperbolic_results['val_auroc'] - euclidean_results['val_auroc'],
        'test_auroc_diff': hyperbolic_results['test_auroc'] - euclidean_results['test_auroc'],
        'tree_corr_diff': hyperbolic_results['tree_distance_spearman'] - euclidean_results['tree_distance_spearman'],
        'norm_depth_diff': hyperbolic_results['norm_depth_spearman'] - euclidean_results['norm_depth_spearman'],
        'hyperbolic_wins': 0
    }

    # Count wins
    if comparison['test_auroc_diff'] > 0.05:
        comparison['hyperbolic_wins'] += 1
    if comparison['tree_corr_diff'] > 0.10:
        comparison['hyperbolic_wins'] += 1
    if hyperbolic_results['norm_depth_spearman'] > 0.5 and euclidean_results['norm_depth_spearman'] < 0.3:
        comparison['hyperbolic_wins'] += 1

    return comparison


if __name__ == '__main__':
    # Test evaluation
    from data import TreeDataset
    from models import EuclideanEmbedding, HyperbolicEmbedding
    import torch

    print("Testing evaluation...")

    # Generate small dataset
    dataset = TreeDataset(num_nodes=100, branching_factor=2, depth=4,
                          embedding_dim=16, seed=42)
    dataset.generate_balanced_tree()
    splits = dataset.get_splits()

    # Test Euclidean model
    print("\n=== Testing Euclidean ===")
    eucl_model = EuclideanEmbedding(dataset.num_nodes, dim=8)
    results = evaluate_model(
        eucl_model, dataset,
        {'val_edges': splits['val_edges'], 'val_non_edges': splits['val_non_edges'],
         'test_edges': splits['test_edges'], 'test_non_edges': splits['test_non_edges']},
        batch_size=64
    )
    print_results(results, "Euclidean (untrained)")

    print("\nEvaluation test completed!")
