"""Embedding models for Euclidean and Hyperbolic spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanEmbedding(nn.Module):
    """Baseline Euclidean embedding model."""

    def __init__(self, num_nodes: int, dim: int, init_scale: float = 1.0):
        """
        Initialize Euclidean embedding.

        Args:
            num_nodes: Number of nodes to embed
            dim: Embedding dimension
            init_scale: Scale for initialization
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.embedding = nn.Embedding(num_nodes, dim)

        # Initialize with reasonable values - not too small to have meaningful distances
        nn.init.normal_(self.embedding.weight, mean=0, std=init_scale)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Get embeddings for given indices."""
        return self.embedding(idx)

    def distance(self, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance between node embeddings.

        Args:
            u_idx: Source node indices
            v_idx: Target node indices

        Returns:
            Euclidean distances
        """
        u = self.embedding(u_idx)
        v = self.embedding(v_idx)
        return torch.norm(u - v, dim=-1)

    def get_embeddings(self) -> torch.Tensor:
        """Get all embeddings."""
        return self.embedding.weight.data


class HyperbolicEmbedding(nn.Module):
    """Poincaré ball embedding model with manual implementation."""

    def __init__(self, num_nodes: int, dim: int, c: float = -1.0,
                 init_scale: float = 1.0, node_depths: torch.Tensor = None):
        """
        Initialize Hyperbolic embedding in Poincaré ball.

        Args:
            num_nodes: Number of nodes to embed
            dim: Embedding dimension
            c: Curvature of the manifold (negative for hyperbolic)
            init_scale: Scale for tangent space initialization
            node_depths: Optional tensor of node depths for hierarchical initialization
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.c = c

        # Maximum norm to stay away from boundary (for unit ball)
        self.max_norm = 0.9

        # Initialize directly in Poincaré ball (inside unit ball)
        self.embedding = nn.Embedding(num_nodes, dim)

        # Depth-aware initialization for better hierarchical structure
        if node_depths is not None:
            self._depth_aware_init(node_depths, init_scale)
        else:
            # Standard random initialization with larger spread for meaningful distances
            nn.init.uniform_(self.embedding.weight, -init_scale * 0.5, init_scale * 0.5)
            # Clip initial embeddings to be well within the ball
            with torch.no_grad():
                norms = torch.norm(self.embedding.weight, dim=-1, keepdim=True)
                self.embedding.weight.data = torch.where(
                    norms > self.max_norm,
                    self.embedding.weight.data / norms * self.max_norm,
                    self.embedding.weight.data
                )

    def _depth_aware_init(self, node_depths: torch.Tensor, init_scale: float):
        """Initialize embeddings with norms proportional to tree depth."""
        max_depth = node_depths.max().item()
        if max_depth > 0:
            # Target norms: root ~0.1, leaves ~0.85
            target_norms = 0.1 + 0.75 * (node_depths.float() / max_depth)
        else:
            target_norms = torch.ones_like(node_depths.float()) * 0.5

        # Initialize randomly and then scale to target norms
        with torch.no_grad():
            # Random initialization
            self.embedding.weight.normal_(mean=0, std=init_scale * 0.1)
            # Normalize and scale to target norms
            norms = torch.norm(self.embedding.weight, dim=-1, keepdim=True)
            self.embedding.weight.data = self.embedding.weight.data / (norms + 1e-8) * target_norms.unsqueeze(-1)

    def _poincare_dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Poincaré ball distance.

        Formula: dist(u, v) = arccosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))

        Args:
            u: Points in Poincaré ball, shape (batch, dim)
            v: Points in Poincaré ball, shape (batch, dim)

        Returns:
            Distances, shape (batch,)
        """
        # Compute norms squared
        u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp_max(1 - 1e-5)
        v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp_max(1 - 1e-5)

        # Compute squared Euclidean distance
        diff = u - v
        diff_norm_sq = torch.sum(diff * diff, dim=-1, keepdim=True)

        # Compute hyperbolic distance
        # Clamp to avoid numerical issues
        alpha = 1 + 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq))
        alpha = alpha.clamp(1 + 1e-5, 1e8)  # arccosh needs input >= 1

        # For hyperbolic distance, we need to scale by sqrt(|c|)
        dist = torch.arccosh(alpha) / (abs(self.c) ** 0.5)

        return dist.squeeze(-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for given indices.

        Args:
            idx: Node indices

        Returns:
            Embeddings in Poincaré ball
        """
        return self.embedding(idx)

    def distance(self, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between node embeddings.

        Args:
            u_idx: Source node indices
            v_idx: Target node indices

        Returns:
            Hyperbolic distances
        """
        u = self.embedding(u_idx)
        v = self.embedding(v_idx)
        return self._poincare_dist(u, v)

    def get_embeddings(self) -> torch.Tensor:
        """Get all embeddings."""
        return self.embedding.weight.data

    def clip_embeddings(self):
        """
        Clip embeddings to stay within Poincaré ball constraints.
        Should be called after each optimization step.
        """
        with torch.no_grad():
            norms = torch.norm(self.embedding.weight.data, dim=-1, keepdim=True)
            clip_mask = norms > self.max_norm
            self.embedding.weight.data = torch.where(
                clip_mask,
                self.embedding.weight.data / norms * self.max_norm,
                self.embedding.weight.data
            )


class DistanceLoss(nn.Module):
    """Contrastive loss for link prediction based on distances."""

    def __init__(self, margin: float = 1.0):
        """
        Initialize distance-based contrastive loss with proper separation.

        Args:
            margin: Minimum gap between negative and positive distances
        """
        super().__init__()
        self.margin = margin

    def forward(self, model: nn.Module,
                pos_pairs: torch.Tensor,
                neg_pairs: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss using margin ranking approach.

        We want: neg_dist - pos_dist >= margin
        Loss = max(0, pos_dist - neg_dist + margin)

        Args:
            model: Embedding model with distance method
            pos_pairs: Positive edge pairs (connected nodes)
            neg_pairs: Negative edge pairs (unconnected nodes)

        Returns:
            Loss value
        """
        pos_dist = model.distance(pos_pairs[:, 0], pos_pairs[:, 1])
        neg_dist = model.distance(neg_pairs[:, 0], neg_pairs[:, 1])

        # For each positive, sample a negative (or use batch)
        # We want: neg_dist - pos_dist > margin
        # Equivalently: pos_dist - neg_dist + margin < 0
        # Loss = max(0, pos_dist - neg_dist + margin)

        # Average over the batch
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0).mean()

        return loss


def create_model(model_type: str, num_nodes: int, dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create embedding models.

    Args:
        model_type: 'euclidean' or 'hyperbolic'
        num_nodes: Number of nodes
        dim: Embedding dimension
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    if model_type.lower() == 'euclidean':
        return EuclideanEmbedding(num_nodes, dim, **kwargs)
    elif model_type.lower() == 'hyperbolic':
        return HyperbolicEmbedding(num_nodes, dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test models
    print("Testing Euclidean model...")
    eucl_model = EuclideanEmbedding(100, 16)
    idx = torch.LongTensor([0, 1, 2, 3])
    emb = eucl_model(idx)
    print(f"  Embeddings shape: {emb.shape}")
    print(f"  Distance self: {eucl_model.distance(torch.tensor([0]), torch.tensor([0]))}")

    print("\nTesting Hyperbolic model...")
    try:
        hyp_model = HyperbolicEmbedding(100, 16)
        emb = hyp_model(idx)
        print(f"  Embeddings shape: {emb.shape}")
        print(f"  Embeddings norm: {torch.norm(emb, dim=-1)}")
        print(f"  Distance self: {hyp_model.distance(torch.tensor([0]), torch.tensor([0]))}")
        print("  Model created successfully!")
    except ImportError as e:
        print(f"  Import error: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
