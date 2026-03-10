from .models import FractionalHedonicGame, Partition
from typing import Set, List, Optional

class StabilityAnalyzer:
    """
    Analyzing stability conditions in Fractional Hedonic Games.
    """
    def __init__(self, game: FractionalHedonicGame):
        self.game = game

    def is_nash_stable(self, partition: Partition) -> bool:
        """
        Nash Stability (NS): No player i can improve their utility by 
        unilaterally moving to another (possibly empty) coalition.
        """
        for i in range(self.game.n):
            current_u = partition.get_player_utility(i)
            
            # Check moving to an existing coalition
            for c_idx, coalition in enumerate(partition.coalitions):
                if i in coalition: continue
                new_u = self.game.get_utility(i, coalition | {i})
                if new_u > current_u:
                    return False
            
            # Check moving to an empty coalition (singleton)
            if current_u < 0:
                return False
                
        return True

    def is_individually_stable(self, partition: Partition) -> bool:
        """
        Individual Stability (IS): No player i can improve their utility by 
        moving to another coalition S without making anyone in S worse off.
        """
        for i in range(self.game.n):
            current_u = partition.get_player_utility(i)
            
            # Existing coalitions
            for c_idx, coalition in enumerate(partition.coalitions):
                if i in coalition: continue
                new_u = self.game.get_utility(i, coalition | {i})
                
                # Player i wants to move
                if new_u > current_u:
                    # Target coalition members must not be worse off
                    is_welcoming = True
                    for member in coalition:
                        if self.game.get_utility(member, coalition | {i}) < self.game.get_utility(member, coalition):
                            is_welcoming = False
                            break
                    if is_welcoming:
                        return False
            
            # Empty coalition
            if current_u < 0: return False
            
        return True

    def is_contractual_individually_stable(self, partition: Partition) -> bool:
        """
        CIS: IS plus the current coalition of i must not be made worse off by i's departure.
        """
        for i in range(self.game.n):
            current_u = partition.get_player_utility(i)
            c_idx_i = partition.player_to_coalition[i]
            current_coalition = partition.coalitions[c_idx_i]
            
            # Existing coalitions
            for c_idx, target_coalition in enumerate(partition.coalitions):
                if i in target_coalition: continue
                new_u = self.game.get_utility(i, target_coalition | {i})
                
                if new_u > current_u:
                    # Outgoing check
                    is_permitted_exit = True
                    if len(current_coalition) > 1:
                        remaining_coalition = current_coalition - {i}
                        for member in remaining_coalition:
                            if self.game.get_utility(member, remaining_coalition) < self.game.get_utility(member, current_coalition):
                                is_permitted_exit = False
                                break
                    
                    if not is_permitted_exit: continue

                    # Welcoming check
                    is_welcoming = True
                    for member in target_coalition:
                        if self.game.get_utility(member, target_coalition | {i}) < self.game.get_utility(member, target_coalition):
                            is_welcoming = False
                            break
                    
                    if is_welcoming: return False
            
            # Empty coalition
            if current_u < 0:
                 # In CIS, singleton move needs permission from current coalition.
                 is_permitted_exit = True
                 if len(current_coalition) > 1:
                    remaining_coalition = current_coalition - {i}
                    for member in remaining_coalition:
                        if self.game.get_utility(member, remaining_coalition) < self.game.get_utility(member, current_coalition):
                            is_permitted_exit = False
                            break
                 if is_permitted_exit: return False
            
        return True

    def is_core_stable(self, partition: Partition) -> bool:
        """
        Core Stability: No subset of players S can form a coalition such that 
        everyone in S is strictly better off than in their current partition.
        Note: This is computationally hard (exponential subsets). 
        Implementation requires specialized algorithms or heuristics for large N.
        """
        # Placeholder: This will be implemented using a recursive search or solver.
        raise NotImplementedError("Core Stability verification requires exponential check.")
