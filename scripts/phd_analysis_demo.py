import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fhg.models import AdditivelySeparableHedonicGame, FractionalHedonicGame
from fhg.optimization import ColumnGenerationSolver
from fhg.report import ResearchReport
from fhg.utils import random_fhg

def run_phd_demo():
    print("Initializing Research Environment...")
    n = 12
    # Generate a random symmetric valuation matrix
    vals = np.random.rand(n, n)
    vals = (vals + vals.T) / 2
    
    # 1. Instantiate an Additively Separable Hedonic Game (ASHG)
    # This is a different class of games than FHG
    game = AdditivelySeparableHedonicGame(vals)
    
    # 2. Use the PhD-level Column Generation Solver
    # This solves the welfare maximization problem using Dantzig-Wolfe decomposition
    print("Solving for Social Welfare via Column Generation...")
    cg_solver = ColumnGenerationSolver(game)
    partition, welfare = cg_solver.solve(max_iter=100)
    
    # 3. Generate a Formal Research Report
    report = ResearchReport(game, partition)
    report.generate_summary()

if __name__ == "__main__":
    run_phd_demo()
