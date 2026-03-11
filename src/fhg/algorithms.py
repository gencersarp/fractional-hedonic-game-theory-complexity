import random
from typing import List, Set, Dict, Tuple, Optional
from .models import FractionalHedonicGame, Partition
from .stability import StabilityAnalyzer

class SearchAlgorithm:
    """
    Algorithms to find stable partitions in FHGs.
    """
    def __init__(self, game: FractionalHedonicGame):
        self.game = game
        self.analyzer = StabilityAnalyzer(game)

    def improve_partition(self, initial_partition: Partition, max_steps: int = 1000) -> Partition:
        """
        Local search: perform improving moves until Nash Stability is reached or max_steps.
        """
        current_partition = initial_partition
        for _ in range(max_steps):
            improving_move = self._find_improving_move(current_partition)
            if not improving_move:
                return current_partition
            
            player, target_c_idx = improving_move
            current_partition = self._apply_move(current_partition, player, target_c_idx)
        
        return current_partition

    def _find_improving_move(self, partition: Partition) -> Optional[Tuple[int, int]]:
        """Finds a player and a target coalition index that improves their utility."""
        players = list(range(self.game.n))
        random.shuffle(players)
        
        for i in players:
            current_u = partition.get_player_utility(i)
            
            # Check all coalitions
            target_indices = list(range(len(partition.coalitions)))
            random.shuffle(target_indices)
            
            for c_idx in target_indices:
                if i in partition.coalitions[c_idx]: continue
                new_u = self.game.get_utility(i, partition.coalitions[c_idx] | {i})
                if new_u > current_u:
                    return i, c_idx
            
            # Check moving to a new singleton coalition
            if current_u < 0:
                return i, -1 # Signal for new coalition
        
        return None

    def _apply_move(self, partition: Partition, player: int, target_c_idx: int) -> Partition:
        new_coalitions = [set(c) for c in partition.coalitions]
        source_c_idx = partition.player_to_coalition[player]
        
        # Remove from source
        new_coalitions[source_c_idx].remove(player)
        
        # Add to target
        if target_c_idx == -1:
            new_coalitions.append({player})
        else:
            new_coalitions[target_c_idx].add(player)
            
        # Clean up empty coalitions
        new_coalitions = [c for c in new_coalitions if len(c) > 0]
        
        return Partition(self.game, new_coalitions)

    def random_restart_search(self, iterations: int = 10) -> Optional[Partition]:
        """
        Runs local search from different random starting points.
        """
        for _ in range(iterations):
            # Generate random partition
            initial_coalitions = self._generate_random_partition()
            p = Partition(self.game, initial_coalitions)
            stable_p = self.improve_partition(p)
            if self.analyzer.is_nash_stable(stable_p):
                return stable_p
        return None

    def _generate_random_partition(self) -> List[Set[int]]:
        players = list(range(self.game.n))
        random.shuffle(players)
        num_coalitions = random.randint(1, self.game.n)
        
        coalitions = [set() for _ in range(num_coalitions)]
        for p in players:
            c_idx = random.randint(0, num_coalitions - 1)
            coalitions[c_idx].add(p)
            
        return [c for c in coalitions if len(c) > 0]

class SocialWelfareSolver:
    """
    Heuristic solvers to maximize social welfare in FHGs.
    Note: Social Welfare Maximization in FHGs is NP-hard.
    """
    def __init__(self, game: FractionalHedonicGame):
        self.game = game

    def simulated_annealing(self, 
                            iterations: int = 5000, 
                            initial_temp: float = 100.0, 
                            cooling_rate: float = 0.99) -> Partition:
        """
        Maximizes social welfare using simulated annealing.
        """
        # Start with a random partition
        search = SearchAlgorithm(self.game)
        current_p = Partition(self.game, search._generate_random_partition())
        current_welfare = current_p.total_social_welfare()
        
        best_p = current_p
        best_welfare = current_welfare
        
        temp = initial_temp
        
        for i in range(iterations):
            # Proposal: move a random player to a random coalition
            player = random.randint(0, self.game.n - 1)
            target_c_idx = random.randint(-1, len(current_p.coalitions) - 1)
            
            # Avoid moving to own coalition
            if target_c_idx != -1 and player in current_p.coalitions[target_c_idx]:
                continue
                
            next_p = search._apply_move(current_p, player, target_c_idx)
            next_welfare = next_p.total_social_welfare()
            
            # Acceptance probability
            delta = next_welfare - current_welfare
            if delta > 0 or random.random() < np.exp(delta / temp):
                current_p = next_p
                current_welfare = next_welfare
                
                if current_welfare > best_welfare:
                    best_p = current_p
                    best_welfare = current_welfare
            
            temp *= cooling_rate
            if temp < 1e-4: break
            
        return best_p
