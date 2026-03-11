import numpy as np
from typing import List, Optional
from .models import FractionalHedonicGame, Partition
from .stability import StabilityAnalyzer
from .algorithms import SearchAlgorithm, SocialWelfareSolver

class StabilityAnalysisSuite:
    """
    Advanced metrics: Price of Anarchy, Price of Stability, 
    and Convergence Cycle Detection.
    """
    def __init__(self, game: FractionalHedonicGame):
        self.game = game
        self.analyzer = StabilityAnalyzer(game)
        self.search = SearchAlgorithm(game)
        self.sw_solver = SocialWelfareSolver(game)

    def calculate_poa_pos(self, trials: int = 20) -> dict:
        """
        Estimates Price of Anarchy (PoA) and Price of Stability (PoS).
        PoA = (Optimal Welfare) / (Worst Nash Welfare)
        PoS = (Optimal Welfare) / (Best Nash Welfare)
        """
        # 1. Estimate Social Optimum
        opt_p = self.sw_solver.simulated_annealing(iterations=10000)
        opt_welfare = opt_p.total_social_welfare()
        if opt_welfare <= 0: return {"PoA": 1.0, "PoS": 1.0}

        # 2. Find many Nash Stable partitions
        nash_welfares = []
        for _ in range(trials):
            stable_p = self.search.random_restart_search(iterations=1)
            if stable_p:
                nash_welfares.append(stable_p.total_social_welfare())
        
        if not nash_welfares:
            return {"error": "No Nash stable partitions found"}

        worst_nash = min(nash_welfares)
        best_nash = max(nash_welfares)

        return {
            "OptimalWelfare": opt_welfare,
            "WorstNash": worst_nash,
            "BestNash": best_nash,
            "PoA": opt_welfare / worst_nash if worst_nash > 0 else float('inf'),
            "PoS": opt_welfare / best_nash if best_nash > 0 else float('inf')
        }

    def detect_cycling(self, max_steps: int = 1000) -> bool:
        """
        Detects if local search can enter an infinite cycle (common in FHGs).
        """
        history = set()
        current_p = Partition(self.game, self.search._generate_random_partition())
        
        for _ in range(max_steps):
            # Canonical representation of partition for history tracking
            canon = tuple(sorted([tuple(sorted(list(c))) for c in current_p.coalitions]))
            if canon in history:
                return True # Cycle detected
            history.add(canon)
            
            move = self.search._find_improving_move(current_p)
            if not move: return False # Converged
            
            current_p = self.search._apply_move(current_p, *move)
            
        return False
