import numpy as np
import networkx as nx
from .models import FractionalHedonicGame

def random_fhg(n: int, symmetric: bool = True, density: float = 0.5) -> FractionalHedonicGame:
    """Generates a random FHG with values in [0, 1]."""
    vals = np.random.rand(n, n)
    if symmetric:
        vals = (vals + vals.T) / 2
    
    # Apply density mask
    mask = np.random.rand(n, n) < density
    vals = vals * mask
    np.fill_diagonal(vals, 0.0)
    
    return FractionalHedonicGame(vals)

def graph_to_fhg(G: nx.Graph) -> FractionalHedonicGame:
    """Converts a NetworkX graph into an FHG with binary (0/1) valuations."""
    adj = nx.to_numpy_array(G)
    return FractionalHedonicGame(adj)

def generate_benchmarks(n_players: int = 10):
    """Factory for standard graph topologies for FHG analysis."""
    topologies = {
        "erdos_renyi": nx.erdos_renyi_graph(n_players, 0.5),
        "barabasi_albert": nx.barabasi_albert_graph(n_players, 3) if n_players > 3 else nx.complete_graph(n_players),
        "cycle": nx.cycle_graph(n_players),
        "complete": nx.complete_graph(n_players),
        "star": nx.star_graph(n_players - 1)
    }
    
    return {name: graph_to_fhg(G) for name, G in topologies.items()}
