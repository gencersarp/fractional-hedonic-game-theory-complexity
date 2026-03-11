import numpy as np
from typing import List, Set, Dict, Optional, Tuple

class FractionalHedonicGame:
    """
    Represents a Fractional Hedonic Game (FHG).
    
    A set of players N = {1, ..., n} and a valuation matrix V = (v_ij),
    where v_ij is the value player i assigns to player j.
    The utility of player i in coalition S (containing i) is:
    u_i(S) = sum_{j in S \ {i}} v_ij / |S|
    """
    def __init__(self, valuations: np.ndarray, names: Optional[List[str]] = None):
        self.n = valuations.shape[0]
        if valuations.shape != (self.n, self.n):
            raise ValueError("Valuation matrix must be square (n x n).")
        
        # Ensure diagonal is zero (no self-loops in utility)
        self.valuations = valuations.copy()
        np.fill_diagonal(self.valuations, 0.0)
        
        self.names = names if names else [f"P{i}" for i in range(self.n)]
        
    def get_utility(self, player_idx: int, coalition: Set[int]) -> float:
        """Calculate the fractional utility of player i in a given coalition."""
        if player_idx not in coalition:
            return -float('inf')
        
        if len(coalition) <= 1:
            return 0.0
            
        sum_vals = sum(self.valuations[player_idx, j] for j in coalition if j != player_idx)
        return sum_vals / len(coalition)

    def is_symmetric(self) -> bool:
        """Check if the valuation matrix is symmetric (v_ij = v_ji)."""
        return np.allclose(self.valuations, self.valuations.T)

class Partition:
    """
    Represents a partition (coalition structure) of the players.
    """
    def __init__(self, game: FractionalHedonicGame, coalitions: List[Set[int]]):
        self.game = game
        self.coalitions = coalitions
        self._validate_partition()
        self.player_to_coalition = {}
        for idx, coalition in enumerate(self.coalitions):
            for player in coalition:
                self.player_to_coalition[player] = idx

    def _validate_partition(self):
        all_players = set()
        for s in self.coalitions:
            for p in s:
                if p in all_players:
                    raise ValueError(f"Player {p} is in multiple coalitions.")
                all_players.add(p)
        if len(all_players) != self.game.n:
             # It's possible to have an incomplete partition in some contexts, 
             # but usually we want all players partitioned.
             pass

    def get_player_utility(self, player_idx: int) -> float:
        c_idx = self.player_to_coalition.get(player_idx)
        if c_idx is None: return 0.0
        return self.game.get_utility(player_idx, self.coalitions[c_idx])

    def total_social_welfare(self) -> float:
        """Sum of utilities of all players."""
        return sum(self.get_player_utility(i) for i in range(self.game.n))

    def average_utility(self) -> float:
        return self.total_social_welfare() / self.game.n

    def __repr__(self):
        return f"Partition({[list(c) for c in self.coalitions]})"
