import numpy as np
from typing import Set, List, Optional
from .models import HedonicGame, FractionalHedonicGame

class GameAxiomVerifier:
    """
    Formal verification of mathematical axioms for Hedonic Games.
    """
    def __init__(self, game: HedonicGame):
        self.game = game

    def is_symmetric(self) -> bool:
        """v_ij = v_ji for all i, j."""
        if not hasattr(self.game, 'valuations'): return False
        vals = self.game.valuations
        return np.allclose(vals, vals.T)

    def is_top_cohesive(self) -> bool:
        """
        A game is top-cohesive if for every subset S, there is a sub-coalition 
        T subset of S such that all members of T prefer T over any other 
        coalition within S. (Condition for Core existence).
        """
        # Note: This check is O(2^n).
        # For our purposes, we'll implement a restricted check.
        # This is often a PhD research question itself.
        raise NotImplementedError("Exponential property check.")

    def satisfies_symmetry_utility(self) -> bool:
        """u_i(S) = u_j(S) for all i, j in S."""
        # Check all possible coalitions (small n)
        n = self.game.n
        from itertools import combinations
        players = list(range(n))
        for r in range(1, n + 1):
            for subset in combinations(players, r):
                S = set(subset)
                if len(S) < 2: continue
                utilities = [self.game.get_utility(i, S) for i in S]
                if not all(np.isclose(utilities[0], u) for u in utilities):
                    return False
        return True

    def has_dummy_player(self) -> Optional[int]:
        """A dummy player i derives 0 value from everyone and everyone derives 0 from i."""
        if not hasattr(self.game, 'valuations'): return None
        for i in range(self.game.n):
            if np.allclose(self.game.valuations[i, :], 0) and np.allclose(self.game.valuations[:, i], 0):
                return i
        return None
