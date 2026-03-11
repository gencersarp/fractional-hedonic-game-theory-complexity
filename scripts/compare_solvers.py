import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fhg.utils import random_fhg
from fhg.optimization import ExactSocialWelfareSolver
from fhg.algorithms import SocialWelfareSolver
from fhg.fairness import FairnessMetrics

def compare_solvers(n_players=10, trials=3):
    print(f"--- Heuristic vs Exact Solver (N={n_players}, Trials={trials}) ---")
    print(f"{'Method':<15} | {'Welfare':<10} | {'Gini':<8} | {'Envy-Free%'}")
    print("-" * 55)
    
    for _ in range(trials):
        game = random_fhg(n_players, density=0.4)
        
        # 1. Exact ILP
        exact_solver = ExactSocialWelfareSolver(game)
        p_exact, w_exact = exact_solver.solve()
        
        # 2. Simulated Annealing (Heuristic)
        heuristic_solver = SocialWelfareSolver(game)
        p_heur = heuristic_solver.simulated_annealing(iterations=2000)
        w_heur = p_heur.total_social_welfare()
        
        # Metrics
        fair = FairnessMetrics()
        
        print(f"{'ILP (Exact)':<15} | {w_exact:<10.2f} | {fair.gini_coefficient(p_exact):<8.3f} | {fair.envy_freeness_degree(p_exact):.1f}%")
        print(f"{'SimAnn (Heur)':<15} | {w_heur:<10.2f} | {fair.gini_coefficient(p_heur):<8.3f} | {fair.envy_freeness_degree(p_heur):.1f}%")
        print("-" * 55)

if __name__ == "__main__":
    # Small N because ILP is exponential in terms of coalitions (2^N)
    compare_solvers(n_players=10, trials=3)
