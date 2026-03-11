# Fractional Hedonic Game Theory: Computational Complexity & Stability

A PhD-level implementation and analysis tool for **Fractional Hedonic Games (FHGs)**. This repository provides a rigorous framework for studying the computational complexity of coalition formation under fractional utility models.

## Theoretical Overview

In a **Fractional Hedonic Game**, we have a set of players $N = \{1, \dots, n\}$ and a valuation matrix $V = (v_{ij})_{i,j \in N}$, where $v_{ij}$ is the value player $i$ derives from player $j$. The utility of player $i$ in a coalition $S \subseteq N$ ($i \in S$) is defined as:

$$u_i(S) = \frac{\sum_{j \in S \setminus \{i\}} v_{ij}}{|S|}$$

This fractional model is particularly interesting because, unlike standard hedonic games, it considers the "average" value, making it scale-invariant but computationally challenging.

### Stability Concepts

This project implements verifiers for the following solution concepts:

1.  **Nash Stability (NS)**: A partition is Nash stable if no player can strictly improve their utility by unilaterally moving to an existing (or empty) coalition.
    *   *Complexity*: Finding a Nash stable partition in general FHGs is PLS-complete.
2.  **Individual Stability (IS)**: A move is permitted only if the player improves AND the members of the target coalition do not lose utility.
3.  **Contractual Individual Stability (CIS)**: Adds the constraint that the current coalition must also not be worse off by the player's departure.
4.  **Core Stability**: A partition is in the core if no subset of players $S \subseteq N$ can form a new coalition such that everyone in $S$ is strictly better off.
    *   *Complexity*: Verification is co-NP-complete; existence is NP-hard.
5.  **Coalition-Proof Nash Equilibrium (CPNE)**: A recursive refinement of Nash Stability where only "self-enforcing" deviations are considered. This prevents deviations that are themselves unstable.
6.  **Altruistic FHGs**: Models where players care about the average utility of their coalition members, governed by an altruism coefficient $\alpha$.

## Advanced Features

-   **Social Welfare Maximization**: Uses Simulated Annealing to find partitions that maximize the sum of utilities (NP-hard).
-   **Phase Transition Analysis**: A script to analyze how graph density $p$ affects the existence and reachability of Nash stable partitions.
-   **Recursive Stability**: Implementation of Bernheim's CPNE definition for small-scale games.

## Project Structure

-   `src/fhg/models.py`: Core data structures for games and partitions.
-   `src/fhg/stability.py`: Rigorous verification of stability conditions.
-   `src/fhg/algorithms.py`: Local search and meta-heuristics for finding stable outcomes.
-   `src/fhg/utils.py`: Graph-theoretic generators (Erdős–Rényi, Barabási–Albert, etc.).
-   `scripts/benchmark_convergence.py`: Empirical analysis of stability across network topologies.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from fhg.models import FractionalHedonicGame, Partition
from fhg.stability import StabilityAnalyzer

# Define a symmetric 3-player game
vals = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
])
game = FractionalHedonicGame(vals)
analyzer = StabilityAnalyzer(game)

# Check stability of a partition
p = Partition(game, [{0, 1}, {2}])
print(f"Nash Stable: {analyzer.is_nash_stable(p)}")
```

## Benchmarking

Run the convergence benchmark to see how stability varies with graph topology:

```bash
python scripts/benchmark_convergence.py
```

## Contributing

This project is designed for researchers in algorithmic game theory. Contributions regarding ILP formulations for core stability or approximation algorithms for social welfare maximization are welcome.
