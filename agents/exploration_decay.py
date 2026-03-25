"""Inventory-aware exploration decay: exploration decreases as inventory depletes."""
import numpy as np


class InventoryAwareExploration:
    """
    Decays exploration as inventory depletes.
    alpha_t = alpha_0 * (inventory_remaining / total_quantity) ^ beta

    Rationale: You can afford to explore early (lots of inventory left, many decisions ahead).
    As inventory depletes, you MUST exploit -- wrong decisions near completion are costly
    and there's no future data to learn from.
    """

    def __init__(self, alpha_0: float = 1.0, beta: float = 0.5, min_alpha: float = 0.05):
        self.alpha_0 = alpha_0
        self.beta = beta
        self.min_alpha = min_alpha

    def get_alpha(self, inventory_fraction: float) -> float:
        """
        Compute current exploration parameter based on inventory remaining.

        Args:
            inventory_fraction: fraction of inventory remaining in [0, 1].

        Returns:
            Decayed alpha value, never below min_alpha.
        """
        inv_frac = max(0.0, min(1.0, inventory_fraction))
        alpha = self.alpha_0 * (inv_frac ** self.beta)
        return max(self.min_alpha, alpha)

    def get_alpha_with_time(self, inventory_fraction: float, time_fraction: float) -> float:
        """
        Combined inventory + time decay.
        Uses the minimum of inventory-based and time-based decay.
        This prevents over-exploration when either resource is scarce.

        Args:
            inventory_fraction: fraction of inventory remaining in [0, 1].
            time_fraction: fraction of time remaining in [0, 1].

        Returns:
            Decayed alpha value, never below min_alpha.
        """
        inv_frac = max(0.0, min(1.0, inventory_fraction))
        time_frac = max(0.0, min(1.0, time_fraction))
        inv_alpha = self.alpha_0 * (inv_frac ** self.beta)
        time_alpha = self.alpha_0 * (time_frac ** self.beta)
        alpha = min(inv_alpha, time_alpha)
        return max(self.min_alpha, alpha)
