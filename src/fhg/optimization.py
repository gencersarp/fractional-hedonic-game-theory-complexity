import pulp
import numpy as np
from typing import List, Set, Tuple
from .models import FractionalHedonicGame, Partition
from itertools import chain, combinations

class ExactSocialWelfareSolver:
    """
    ILP formulation for Social Welfare Maximization in FHGs.
    Warning: The number of possible coalitions is 2^n - 1.
    This formulation is feasible only for small N (e.g., N <= 15).
    """
    def __init__(self, game: FractionalHedonicGame):
        self.game = game

    def solve(self) -> Tuple[Partition, float]:
        """
        Solves: maximize sum_{i in N} u_i(S_i)
        Subject to: sum_{S contains i} x_S = 1 for all i
        """
        n = self.game.n
        players = list(range(n))
        
        # Generate all non-empty coalitions
        all_coalitions = []
        for r in range(1, n + 1):
            for subset in combinations(players, r):
                all_coalitions.append(set(subset))
        
        # Binary variables for each coalition
        # x[k] = 1 if coalition k is in the partition
        prob = pulp.LpProblem("SocialWelfareMaximization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(len(all_coalitions)), cat=pulp.LpBinary)
        
        # Objective Function: Sum of player utilities in selected coalitions
        # u(S) = sum_{i in S} u_i(S)
        welfare_coeffs = []
        for S in all_coalitions:
            total_u_S = sum(self.game.get_utility(i, S) for i in S)
            welfare_coeffs.append(total_u_S)
            
        prob += pulp.lpSum([welfare_coeffs[k] * x[k] for k in range(len(all_coalitions))])
        
        # Constraint: Each player must be in exactly one coalition
        for i in range(n):
            relevant_coalitions = [k for k, S in enumerate(all_coalitions) if i in S]
            prob += pulp.lpSum([x[k] for k in relevant_coalitions]) == 1
            
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            return None, 0.0
            
        # Extract partition
        final_coalitions = []
        for k in range(len(all_coalitions)):
            if pulp.value(x[k]) > 0.5:
                final_coalitions.append(all_coalitions[k])
                
        partition = Partition(self.game, final_coalitions)
        return partition, partition.total_social_welfare()
