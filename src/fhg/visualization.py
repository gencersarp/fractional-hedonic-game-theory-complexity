import networkx as nx
import matplotlib.pyplot as plt
from .models import Partition

def visualize_partition(partition: Partition, title: str = "FHG Coalition Structure"):
    """
    Visualizes the partition on the underlying valuation graph.
    """
    game = partition.game
    G = nx.from_numpy_array(game.valuations)
    
    # Generate colors for each coalition
    colors = plt.cm.rainbow(np.linspace(0, 1, len(partition.coalitions)))
    node_colors = {}
    for c_idx, coalition in enumerate(partition.coalitions):
        for player in coalition:
            node_colors[player] = colors[c_idx]
            
    color_list = [node_colors[i] for i in range(game.n)]
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    
    # Draw edges with weight-proportional thickness
    weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.3)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=color_list, node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

import numpy as np # Ensure numpy is available for linspace
