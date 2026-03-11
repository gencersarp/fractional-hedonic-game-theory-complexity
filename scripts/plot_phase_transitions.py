import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fhg.utils import random_fhg
from fhg.algorithms import SearchAlgorithm
from fhg.stability import StabilityAnalyzer
from fhg.models import Partition

def analyze_phase_transition(n_players=10, densities=None, trials_per_density=20):
    if densities is None:
        densities = np.linspace(0.05, 0.95, 10)
    
    success_rates = []
    
    print(f"--- Phase Transition Analysis (N={n_players}) ---")
    
    for d in densities:
        successes = 0
        for _ in range(trials_per_density):
            game = random_fhg(n_players, density=d)
            search = SearchAlgorithm(game)
            analyzer = StabilityAnalyzer(game)
            
            # Start from random
            p = Partition(game, search._generate_random_partition())
            final_p = search.improve_partition(p, max_steps=1000)
            
            if analyzer.is_nash_stable(final_p):
                successes += 1
        
        rate = (successes / trials_per_density) * 100
        success_rates.append(rate)
        print(f"Density: {d:.2f} | Nash Success Rate: {rate:.1f}%")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(densities, success_rates, marker='o', linestyle='-', color='b')
    plt.title(f"Nash Stability Phase Transition (N={n_players})")
    plt.xlabel("Graph Density (p)")
    plt.ylabel("Success Rate of Local Search (%)")
    plt.grid(True)
    plt.savefig("phase_transition.png")
    print("\nPhase transition plot saved as 'phase_transition.png'")

if __name__ == "__main__":
    analyze_phase_transition(n_players=10, trials_per_density=20)
