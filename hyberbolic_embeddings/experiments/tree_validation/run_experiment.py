#!/usr/bin/env python3
"""
Main experiment runner for hyperbolic vs Euclidean embedding comparison.

Usage:
    python run_experiment.py --help
    python run_experiment.py --model-type euclidean --dim 16
    python run_experiment.py --model-type hyperbolic --dim 16 --seed 42
    python run_experiment.py --run-both --dim 16
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import TreeDataset, generate_dataset
from models import EuclideanEmbedding, HyperbolicEmbedding
from train import train_model
from evaluate import evaluate_model, print_results, compare_results, visualize_embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run hyperbolic vs Euclidean embedding experiments on tree data'
    )

    # Model selection
    parser.add_argument('--model-type', type=str, default='euclidean',
                        choices=['euclidean', 'hyperbolic'],
                        help='Type of embedding model')
    parser.add_argument('--run-both', action='store_true',
                        help='Run both Euclidean and Hyperbolic models')

    # Dataset parameters
    parser.add_argument('--num-nodes', type=int, default=1000,
                        help='Number of nodes in tree')
    parser.add_argument('--branching-factor', type=int, default=4,
                        help='Branching factor of tree')
    parser.add_argument('--depth', type=int, default=5,
                        help='Maximum depth of tree')
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Dimension of node features (for data generation)')

    # Model parameters
    parser.add_argument('--dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--curvature', type=float, default=-1.0,
                        help='Curvature for hyperbolic model (negative)')

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=100,
                        help='Evaluate every N epochs')

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Output directory for results')
    parser.add_argument('--data-dir', type=str, default='experiments/tree_validation/data',
                        help='Directory for dataset storage')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_arg: str) -> str:
    """Determine device to use."""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def run_single_experiment(args, model_type: str, dataset: TreeDataset,
                          splits: dict) -> dict:
    """
    Run a single experiment (Euclidean or Hyperbolic).

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Running {model_type.upper()} Experiment")
    print(f"{'='*70}")
    print(f"Seed: {args.seed}")
    print(f"Dimension: {args.dim}")
    print(f"Device: {args.device}")
    print(f"Dataset: {dataset.num_nodes} nodes, {len(dataset.edges)} edges")
    print(f"{'='*70}\n")

    # Create model
    if model_type == 'euclidean':
        model = EuclideanEmbedding(dataset.num_nodes, dim=args.dim)
        lr = args.learning_rate
    elif model_type == 'hyperbolic':
        # Get node depths for hierarchical initialization
        node_depths = torch.LongTensor(dataset.node_depths)
        model = HyperbolicEmbedding(dataset.num_nodes, dim=args.dim, c=args.curvature,
                                    node_depths=node_depths)
        # Use lower learning rate for hyperbolic models (more stable)
        lr = args.learning_rate * 0.5
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Prepare data dictionaries for training
    train_data = {
        'train_edges': splits['train_edges'],
        'train_non_edges': splits['train_non_edges']
    }
    val_data = {
        'val_edges': splits['val_edges'],
        'val_non_edges': splits['val_non_edges']
    }

    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, model_type, f'seed_{args.seed}')
    os.makedirs(model_output_dir, exist_ok=True)

    # Train model
    start_time = time.time()
    history = train_model(
        model,
        train_data,
        val_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=lr,
        device=args.device,
        save_dir=model_output_dir,
        model_name=f'{model_type}_d{args.dim}',
        eval_every=args.eval_every
    )
    train_time = time.time() - start_time

    # Load best model
    checkpoint_path = os.path.join(model_output_dir, f'{model_type}_d{args.dim}_best.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with AUROC {checkpoint['auroc']:.4f}")

    # Evaluate model
    test_data = {
        'val_edges': splits['val_edges'],
        'val_non_edges': splits['val_non_edges'],
        'test_edges': splits['test_edges'],
        'test_non_edges': splits['test_non_edges']
    }

    results = evaluate_model(
        model, dataset, test_data,
        batch_size=args.batch_size,
        device=args.device
    )

    # Add metadata
    results['model_type'] = model_type
    results['dim'] = args.dim
    results['seed'] = args.seed
    results['train_time'] = train_time
    results['num_epochs'] = args.num_epochs
    results['num_nodes'] = dataset.num_nodes
    results['num_edges'] = len(dataset.edges)

    # Print results
    print_results(results, model_type.capitalize())

    # Generate visualizations
    viz_path = os.path.join(model_output_dir, f'{model_type}_embeddings.png')
    try:
        visualize_embeddings(model, dataset, viz_path, device=args.device, method='pca')
        print(f"Saved visualization to {viz_path}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")

    # Save results
    results_path = os.path.join(model_output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in results.items() if 'details' not in k}
        json.dump(json_results, f, indent=2)
    print(f"Saved results to {results_path}")

    return results


def run_comparison_experiment(args):
    """Run both Euclidean and Hyperbolic experiments and compare."""
    print("\n" + "="*70)
    print("HYPERBOLIC vs EUCLIDEAN COMPARISON EXPERIMENT")
    print("="*70)

    # Generate or load dataset
    data_path = os.path.join(args.data_dir, 'tree_data.pkl')

    if os.path.exists(data_path):
        print(f"Loading existing dataset from {data_path}")
        dataset = TreeDataset.load(data_path)
        # Re-generate splits with current seed
        set_seed(args.seed)
        splits = dataset.get_splits()
    else:
        print("Generating new dataset...")
        dataset = generate_dataset(
            save_dir=args.data_dir,
            num_nodes=args.num_nodes,
            branching_factor=args.branching_factor,
            depth=args.depth,
            embedding_dim=args.embedding_dim,
            seed=args.seed
        )
        splits = dataset.get_splits()

    # Run Euclidean experiment
    args.model_type = 'euclidean'
    euclidean_results = run_single_experiment(args, 'euclidean', dataset, splits)

    # Run Hyperbolic experiment
    args.model_type = 'hyperbolic'
    hyperbolic_results = run_single_experiment(args, 'hyperbolic', dataset, splits)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    comparison = compare_results(euclidean_results, hyperbolic_results)

    print(f"\nEuclidean Results:")
    print(f"  Test AUROC:           {euclidean_results['test_auroc']:.4f}")
    print(f"  Tree Dist Correlation: {euclidean_results['tree_distance_spearman']:.4f}")
    print(f"  Norm-Depth Correlation: {euclidean_results['norm_depth_spearman']:.4f}")

    print(f"\nHyperbolic Results:")
    print(f"  Test AUROC:           {hyperbolic_results['test_auroc']:.4f}")
    print(f"  Tree Dist Correlation: {hyperbolic_results['tree_distance_spearman']:.4f}")
    print(f"  Norm-Depth Correlation: {hyperbolic_results['norm_depth_spearman']:.4f}")

    print(f"\nDifferences (Hyperbolic - Euclidean):")
    print(f"  Test AUROC:           {comparison['test_auroc_diff']:+.4f}")
    print(f"  Tree Dist Correlation: {comparison['tree_corr_diff']:+.4f}")
    print(f"  Norm-Depth Correlation: {comparison['norm_depth_diff']:+.4f}")

    # Decision
    print("\n" + "="*70)
    print("DECISION")
    print("="*70)

    if comparison['hyperbolic_wins'] >= 2:
        print("✓ HYPERBOLIC WINS: Clear advantage on >= 2 metrics")
        print("  Recommendation: Proceed with hyperbolic geometry")
    elif comparison['test_auroc_diff'] > 0.02 and comparison['tree_corr_diff'] > 0.05:
        print("~ HYPERBOLIC SLIGHTLY BETTER: Modest improvements")
        print("  Recommendation: Consider scaling up experiment")
    elif abs(comparison['test_auroc_diff']) < 0.02 and abs(comparison['tree_corr_diff']) < 0.05:
        print("~ NO CLEAR DIFFERENCE: Results within noise")
        print("  Recommendation: Try larger dimensions or tree")
    else:
        print("✗ EUCLIDEAN BETTER OR EQUAL: Hyperbolic not showing advantage")
        print("  Recommendation: Debug or reconsider approach")

    print("="*70)

    # Save comparison
    comparison_data = {
        'euclidean': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in euclidean_results.items() if 'details' not in k},
        'hyperbolic': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in hyperbolic_results.items() if 'details' not in k},
        'comparison': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in comparison.items()},
        'config': vars(args)
    }

    comparison_path = os.path.join(args.output_dir, f'comparison_seed{args.seed}.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nSaved comparison to {comparison_path}")

    return euclidean_results, hyperbolic_results, comparison


def main():
    """Main entry point."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    args.device = get_device(args.device)

    print(f"Hyperbolic Validation Experiment - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {args.device}")

    # Run experiments
    if args.run_both:
        run_comparison_experiment(args)
    else:
        # Generate dataset
        data_path = os.path.join(args.data_dir, 'tree_data.pkl')
        if os.path.exists(data_path):
            print(f"Loading existing dataset from {data_path}")
            dataset = TreeDataset.load(data_path)
            set_seed(args.seed)
            splits = dataset.get_splits()
        else:
            dataset = generate_dataset(
                save_dir=args.data_dir,
                num_nodes=args.num_nodes,
                branching_factor=args.branching_factor,
                depth=args.depth,
                embedding_dim=args.embedding_dim,
                seed=args.seed
            )
            splits = dataset.get_splits()

        # Run single experiment
        run_single_experiment(args, args.model_type, dataset, splits)

    print("\n" + "="*70)
    print("Experiment completed!")
    print("="*70)


if __name__ == '__main__':
    main()
