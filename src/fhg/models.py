import numpy as np
from typing import List, Set, Dict, Optional, Tuple
from abc import ABC, abstractmethod

class HedonicGame(ABC):
    """Abstract Base Class for Hedonic Games."""
    def __init__(self, n: int, names: Optional[List[str]] = None):
        self.n = n
        self.names = names if names else [f"P{i}" for i in range(n)]

    @abstractmethod
    def get_utility(self, player_idx: int, coalition: Set[int]) -> float:
        pass

class FractionalHedonicGame(HedonicGame):
    """Standard FHG implementation."""
    def __init__(self, valuations: np.ndarray, names: Optional[List[str]] = None):
        super().__init__(valuations.shape[0], names)
        self.valuations = valuations.copy()
        np.fill_diagonal(self.valuations, 0.0)
        
    def get_utility(self, player_idx: int, coalition: Set[int]) -> float:
        if player_idx not in coalition: return -float('inf')
        if len(coalition) <= 1: return 0.0
        sum_vals = sum(self.valuations[player_idx, j] for j in coalition if j != player_idx)
        return sum_vals / len(coalition)

class AdditivelySeparableHedonicGame(HedonicGame):
    """
    In ASHG, u_i(S) = sum_{j in S} v_ij. 
    Unlike FHG, there is no normalization by |S|.
    """
    def __init__(self, valuations: np.ndarray, names: Optional[List[str]] = None):
        super().__init__(valuations.shape[0], names)
        self.valuations = valuations.copy()
        np.fill_diagonal(self.valuations, 0.0)

    def get_utility(self, player_idx: int, coalition: Set[int]) -> float:
        if player_idx not in coalition: return -float('inf')
        return sum(self.valuations[player_idx, j] for j in coalition)

class AltruisticFHG(FractionalHedonicGame):
    def __init__(self, valuations: np.ndarray, alpha: float = 0.5, names: Optional[List[str]] = None):
        super().__init__(valuations, names)
        self.alpha = alpha

    def get_utility(self, player_idx: int, coalition: Set[int]) -> float:
        if player_idx not in coalition: return -float('inf')
        if len(coalition) <= 1: return 0.0
        u_self = super().get_utility(player_idx, coalition)
        others = [j for j in coalition if j != player_idx]
        u_others = sum(super().get_utility(j, coalition) for j in others) / len(others)
        return (1 - self.alpha) * u_self + self.alpha * u_others

class Partition:
    def __init__(self, game: HedonicGame, coalitions: List[Set[int]]):
        self.game = game
        self.coalitions = coalitions
        self.player_to_coalition = {}
        for idx, coalition in enumerate(self.coalitions):
            for player in coalition:
                self.player_to_coalition[player] = idx

    def get_player_utility(self, player_idx: int) -> float:
        c_idx = self.player_to_coalition.get(player_idx)
        if c_idx is None: return 0.0
        return self.game.get_utility(player_idx, self.coalitions[c_idx])

    def total_social_welfare(self) -> float:
        return sum(self.get_player_utility(i) for i in range(self.game.n))

    def __repr__(self):
        return f"Partition({[list(c) for c in self.coalitions]})"
