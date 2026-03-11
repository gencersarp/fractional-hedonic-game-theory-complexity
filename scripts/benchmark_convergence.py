import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fhg.utils import generate_benchmarks
from fhg.algorithms import SearchAlgorithm, SocialWelfareSolver
from fhg.stability import StabilityAnalyzer
from fhg.analysis import StabilityAnalysisSuite
from fhg.models import Partition

def run_benchmarks(n_players=15, trials=10):
    print(f"--- Advanced FHG Complexity Analysis (N={n_players}) ---")
    print(f"{'Topology':<20} | {'PoA (Est)':<10} | {'PoS (Est)':<10} | {'Cycling?'}")
    print("-" * 65)
    
    benchmarks = generate_benchmarks(n_players)
    
    for name, game in benchmarks.items():
        suite = StabilityAnalysisSuite(game)
        
        # Calculate PoA / PoS
        metrics = suite.calculate_poa_pos(trials=trials)
        
        # Detect potential for cycling
        has_cycles = suite.detect_cycling(max_steps=2000)
        
        poa = f"{metrics.get('PoA', 0):.2f}" if 'PoA' in metrics else "N/A"
        pos = f"{metrics.get('PoS', 0):.2f}" if 'PoS' in metrics else "N/A"
        cycle_str = "YES" if has_cycles else "NO"
        
        print(f"{name:<20} | {poa:<10} | {pos:<10} | {cycle_str}")

if __name__ == "__main__":
    # Reducing trials for a quicker demonstration of complexity analysis
    run_benchmarks(n_players=12, trials=5)
