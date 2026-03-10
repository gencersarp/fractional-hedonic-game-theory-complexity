import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fhg.utils import generate_benchmarks
from fhg.algorithms import SearchAlgorithm
from fhg.stability import StabilityAnalyzer
from fhg.models import Partition

def run_benchmarks(n_players=15, trials=50):
    print(f"--- Benchmarking FHG Convergence (N={n_players}, Trials={trials}) ---")
    print(f"{'Topology':<20} | {'NS Success %':<12} | {'Avg Steps':<10}")
    print("-" * 50)
    
    benchmarks = generate_benchmarks(n_players)
    
    for name, game in benchmarks.items():
        success_count = 0
        total_steps = 0
        
        search = SearchAlgorithm(game)
        analyzer = StabilityAnalyzer(game)
        
        for _ in range(trials):
            # Start from random partition
            initial_coalitions = search._generate_random_partition()
            p = Partition(game, initial_coalitions)
            
            # Local search
            final_p = search.improve_partition(p, max_steps=500)
            
            if analyzer.is_nash_stable(final_p):
                success_count += 1
                # Note: step count is not tracked in improve_partition currently, 
                # but we could add it. For now, we'll just track success.
        
        success_rate = (success_count / trials) * 100
        print(f"{name:<20} | {success_rate:<12.1f} | {'N/A'}")

if __name__ == "__main__":
    run_benchmarks(n_players=15, trials=50)
