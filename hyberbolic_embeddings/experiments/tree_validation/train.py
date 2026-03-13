"""Training script for link prediction on tree data."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional
import time
import os
import json

from models import EuclideanEmbedding, HyperbolicEmbedding, DistanceLoss


def train_epoch(model: torch.nn.Module,
                train_edges: torch.Tensor,
                train_non_edges: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                criterion: DistanceLoss,
                batch_size: int,
                device: str = 'cpu',
                clip_grad_norm: Optional[float] = 1.0) -> float:
    """
    Train for one epoch.

    Args:
        model: The embedding model
        train_edges: Positive training edges
        train_non_edges: Negative training edges
        optimizer: Optimizer
        criterion: Loss function
        batch_size: Batch size
        device: Device to use
        clip_grad_norm: Gradient clipping norm (None for no clipping)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Shuffle training data
    n_edges = len(train_edges)
    n_non_edges = len(train_non_edges)

    # Create batches
    indices_edges = torch.randperm(n_edges)
    indices_non_edges = torch.randperm(n_non_edges)

    batch_size_half = batch_size // 2

    for i in range(0, n_edges, batch_size_half):
        # Sample positive batch
        end_idx = min(i + batch_size_half, n_edges)
        pos_idx = indices_edges[i:end_idx]
        pos_batch = train_edges[pos_idx].to(device)

        # Sample negative batch
        neg_idx = torch.randperm(n_non_edges)[:len(pos_idx)]
        neg_batch = train_non_edges[neg_idx].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss = criterion(model, pos_batch, neg_batch)

        # Backward pass
        loss.backward()

        # Clip gradients
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Optimizer step
        optimizer.step()

        # Clip embeddings for hyperbolic model
        if isinstance(model, HyperbolicEmbedding):
            model.clip_embeddings()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model: torch.nn.Module,
             edges: torch.Tensor,
             non_edges: torch.Tensor,
             batch_size: int,
             device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model on validation/test set.

    Args:
        model: The embedding model
        edges: Positive edges
        non_edges: Negative edges
        batch_size: Batch size for evaluation
        device: Device to use

    Returns:
        Dictionary with metrics including AUROC
    """
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

    # Compute AUROC
    # Create labels (1 for positive, 0 for negative)
    labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    scores = np.concatenate([pos_dists, neg_dists])

    # For AUROC: lower distance should mean higher probability of edge
    # So we use negative distances as scores
    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(labels, -scores)
    except:
        auroc = 0.5  # Default if computation fails

    return {
        'auroc': auroc,
        'pos_dist_mean': float(np.mean(pos_dists)),
        'pos_dist_std': float(np.std(pos_dists)),
        'neg_dist_mean': float(np.mean(neg_dists)),
        'neg_dist_std': float(np.std(neg_dists)),
    }


def train_model(model: torch.nn.Module,
                train_data: Dict,
                val_data: Dict,
                num_epochs: int = 1000,
                batch_size: int = 256,
                learning_rate: float = 1e-3,
                device: str = 'cpu',
                save_dir: str = 'checkpoints',
                model_name: str = 'model',
                eval_every: int = 100) -> Dict:
    """
    Train a model with link prediction task.

    Args:
        model: The embedding model
        train_data: Dictionary with train_edges and train_non_edges
        val_data: Dictionary with val_edges and val_non_edges
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
        model_name: Name prefix for saved models
        eval_every: Evaluate every N epochs

    Returns:
        Dictionary with training history
    """
    # Setup
    model = model.to(device)
    criterion = DistanceLoss(margin=2.0)  # Larger margin for better separation

    # Use Riemannian optimizer for hyperbolic model, Adam for Euclidean
    if isinstance(model, HyperbolicEmbedding):
        try:
            from geoopt.optim import RiemannianAdam
            optimizer = RiemannianAdam(model.parameters(), lr=learning_rate)
        except ImportError:
            print("geoopt.optim.RiemannianAdam not available, using Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'val_auroc': [],
        'val_pos_dist': [],
        'val_neg_dist': []
    }

    best_auroc = 0.0

    print(f"Training {model_name} for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model,
            train_data['train_edges'],
            train_data['train_non_edges'],
            optimizer,
            criterion,
            batch_size,
            device
        )

        history['train_loss'].append(train_loss)

        # Evaluate
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate(
                model,
                val_data['val_edges'],
                val_data['val_non_edges'],
                batch_size,
                device
            )

            history['val_auroc'].append(val_metrics['auroc'])
            history['val_pos_dist'].append(val_metrics['pos_dist_mean'])
            history['val_neg_dist'].append(val_metrics['neg_dist_mean'])

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f} | "
                  f"Pos dist: {val_metrics['pos_dist_mean']:.4f} | "
                  f"Neg dist: {val_metrics['neg_dist_mean']:.4f}")

            # Save best model
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auroc': best_auroc,
                }, checkpoint_path)

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")
    print(f"Best Val AUROC: {best_auroc:.4f}")

    # Save final checkpoint
    checkpoint_path = os.path.join(save_dir, f'{model_name}_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, checkpoint_path)

    # Save history as JSON
    history_path = os.path.join(save_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == '__main__':
    # Test training
    from data import TreeDataset

    print("Testing training...")

    # Generate small dataset
    dataset = TreeDataset(num_nodes=100, branching_factor=2, depth=4,
                          embedding_dim=16, seed=42)
    dataset.generate_balanced_tree()
    splits = dataset.get_splits()

    print(f"Dataset: {dataset.num_nodes} nodes, {len(dataset.edges)} edges")

    # Train Euclidean model
    print("\n=== Euclidean Model ===")
    eucl_model = EuclideanEmbedding(dataset.num_nodes, dim=8)
    eucl_history = train_model(
        eucl_model,
        {'train_edges': splits['train_edges'], 'train_non_edges': splits['train_non_edges']},
        {'val_edges': splits['val_edges'], 'val_non_edges': splits['val_non_edges']},
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        model_name='euclidean_test',
        eval_every=20
    )

    # Train Hyperbolic model
    print("\n=== Hyperbolic Model ===")
    hyp_model = HyperbolicEmbedding(dataset.num_nodes, dim=8)
    hyp_history = train_model(
        hyp_model,
        {'train_edges': splits['train_edges'], 'train_non_edges': splits['train_non_edges']},
        {'val_edges': splits['val_edges'], 'val_non_edges': splits['val_non_edges']},
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        model_name='hyperbolic_test',
        eval_every=20
    )

    print("\nTraining test completed!")
