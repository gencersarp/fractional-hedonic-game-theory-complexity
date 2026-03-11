from .models import Partition, HedonicGame
from .stability import StabilityAnalyzer
from .fairness import FairnessMetrics
from .axioms import GameAxiomVerifier

class ResearchReport:
    """
    Generates a structured research report for a given game and partition.
    """
    def __init__(self, game: HedonicGame, partition: Partition):
        self.game = game
        self.partition = partition
        self.analyzer = StabilityAnalyzer(game)
        self.fairness = FairnessMetrics()
        self.axioms = GameAxiomVerifier(game)

    def generate_summary(self):
        print("=" * 60)
        print("ALGORITHMIC GAME THEORY: HEDONIC GAME ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n[1] GAME PROPERTIES")
        print(f"Game Type: {type(self.game).__name__}")
        print(f"Number of Players: {self.game.n}")
        print(f"Symmetric Valuations: {self.axioms.is_symmetric()}")
        dummy = self.axioms.has_dummy_player()
        print(f"Dummy Player: {'None' if dummy is None else f'P{dummy}'}")
        
        print(f"\n[2] STABILITY ANALYSIS")
        print(f"Nash Stable: {self.analyzer.is_nash_stable(self.partition)}")
        print(f"Individually Stable: {self.analyzer.is_individually_stable(self.partition)}")
        
        print(f"\n[3] WELFARE & FAIRNESS")
        print(f"Total Social Welfare: {self.partition.total_social_welfare():.4f}")
        print(f"Gini Coefficient: {self.fairness.gini_coefficient(self.partition):.4f}")
        print(f"Envy-Freeness: {self.fairness.envy_freeness_degree(self.partition):.2f}%")
        
        print(f"\n[4] COALITION STRUCTURE")
        for i, c in enumerate(self.partition.coalitions):
            print(f"Coalition {i+1}: {sorted(list(c))}")
        print("=" * 60)
