import numpy as np
from .models import Partition

class FairnessMetrics:
    """
    Analyzing the distributional properties of FHG partitions.
    """
    @staticmethod
    def gini_coefficient(partition: Partition) -> float:
        """
        Calculates the Gini coefficient for player utilities in the partition.
        G = 0 means perfect equality (everyone has same utility).
        G = 1 means maximum inequality.
        """
        utilities = [partition.get_player_utility(i) for i in range(partition.game.n)]
        # Shift to non-negative if necessary (FHG can have negative values)
        min_u = min(utilities)
        if min_u < 0:
            utilities = [u - min_u for u in utilities]
            
        if sum(utilities) == 0: return 0.0
        
        utilities = sorted(utilities)
        n = len(utilities)
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * utilities)) / (n * np.sum(utilities)))

    @staticmethod
    def egalitarian_welfare(partition: Partition) -> float:
        """Returns the minimum utility of any player."""
        utilities = [partition.get_player_utility(i) for i in range(partition.game.n)]
        return min(utilities)

    @staticmethod
    def envy_freeness_degree(partition: Partition) -> float:
        """
        Calculates the percentage of players who are envy-free.
        A player i is envy-free if they do not prefer any other coalition S in the partition.
        """
        n = partition.game.n
        envy_free_count = 0
        for i in range(n):
            current_u = partition.get_player_utility(i)
            is_envious = False
            for c_idx, coalition in enumerate(partition.coalitions):
                if i in coalition: continue
                # i prefers the other coalition S | {i}
                if partition.game.get_utility(i, coalition | {i}) > current_u + 1e-9:
                    is_envious = True
                    break
            
            if not is_envious:
                envy_free_count += 1
                
        return (envy_free_count / n) * 100
