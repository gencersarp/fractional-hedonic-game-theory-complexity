import pulp
import numpy as np
from typing import List, Set, Tuple
from .models import FractionalHedonicGame, Partition, HedonicGame
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

class ColumnGenerationSolver:
    """
    Implements a Column Generation (Dantzig-Wolfe) approach.
    RMP: Linear Relaxation of the Set Partitioning Problem.
    Sub-problem: Finding a coalition S with positive reduced cost.
    Reduced Cost(S) = Welfare(S) - sum_{i in S} shadow_price[i]
    """
    def __init__(self, game: HedonicGame):
        self.game = game

    def solve(self, max_iter: int = 50) -> Tuple[Partition, float]:
        n = self.game.n
        # Initial columns: Singletons
        coalitions = [{i} for i in range(n)]
        
        for iteration in range(max_iter):
            # 1. Solve RMP (Linear Relaxation)
            prob = pulp.LpProblem("MasterProblem", pulp.LpMaximize)
            x = pulp.LpVariable.dicts("x", range(len(coalitions)), lowBound=0, upBound=1, cat=pulp.LpContinuous)
            
            welfare_coeffs = [sum(self.game.get_utility(i, S) for i in S) for S in coalitions]
            prob += pulp.lpSum([welfare_coeffs[k] * x[k] for k in range(len(coalitions))])
            
            # Shadow price constraints (dual variables)
            constraints = []
            for i in range(n):
                c = pulp.lpSum([x[k] for k, S in enumerate(coalitions) if i in S]) == 1
                prob += c
                constraints.append(c)
                
            solver = pulp.PULP_CBC_CMD(msg=0)
            prob.solve(solver)
            
            shadow_prices = [constraints[i].pi for i in range(n)]
            
            # 2. Pricing Sub-problem: maximize Reduced Cost
            # This sub-problem is itself hard (often NP-hard), so we use a heuristic or solver.
            new_S, reduced_cost = self._pricing_subproblem(shadow_prices)
            
            if reduced_cost <= 1e-4:
                # No more columns with positive reduced cost found
                break
                
            coalitions.append(new_S)

        # 3. Solve final RMP as Integer Program
        prob = pulp.LpProblem("FinalMasterProblem", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(len(coalitions)), cat=pulp.LpBinary)
        welfare_coeffs = [sum(self.game.get_utility(i, S) for i in S) for S in coalitions]
        prob += pulp.lpSum([welfare_coeffs[k] * x[k] for k in range(len(coalitions))])
        for i in range(n):
            prob += pulp.lpSum([x[k] for k, S in enumerate(coalitions) if i in S]) == 1
        
        prob.solve(solver)
        
        final_coalitions = [coalitions[k] for k in range(len(coalitions)) if pulp.value(x[k]) > 0.5]
        partition = Partition(self.game, final_coalitions)
        return partition, partition.total_social_welfare()

    def _pricing_subproblem(self, shadow_prices: List[float]) -> Tuple[Set[int], float]:
        """
        Sub-problem: argmax_{S subset N} [ Welfare(S) - sum_{i in S} pi_i ]
        We use a greedy heuristic with random restarts to find improving columns.
        """
        best_S = set()
        best_rc = 0.0
        
        for _ in range(50): # Multiple restarts
            S = {random.randint(0, self.game.n - 1)}
            improved = True
            while improved:
                improved = False
                # Try adding or removing players
                for i in range(self.game.n):
                    candidate_S = S ^ {i} # Toggle inclusion
                    if not candidate_S: continue
                    
                    welfare = sum(self.game.get_utility(p, candidate_S) for p in candidate_S)
                    rc = welfare - sum(shadow_prices[p] for p in candidate_S)
                    
                    if rc > best_rc:
                        best_rc = rc
                        best_S = candidate_S
                        S = candidate_S
                        improved = True
                        break
        return best_S, best_rc

import random # for pricing subproblem
