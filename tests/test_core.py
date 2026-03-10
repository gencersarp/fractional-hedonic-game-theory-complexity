import unittest
import numpy as np
from fhg.models import FractionalHedonicGame, Partition
from fhg.stability import StabilityAnalyzer
from fhg.algorithms import SearchAlgorithm

class TestFHG(unittest.TestCase):
    def setUp(self):
        # A simple 3-player symmetric game
        # Players 0 and 1 like each other (value 1.0)
        # Player 2 is neutral
        self.valuations = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        self.game = FractionalHedonicGame(self.valuations)

    def test_utility_calculation(self):
        # Coalition {0, 1}
        # Player 0 utility: 1.0 / 2 = 0.5
        u0 = self.game.get_utility(0, {0, 1})
        self.assertEqual(u0, 0.5)

        # Player 2 utility in {2}: 0.0
        u2 = self.game.get_utility(2, {2})
        self.assertEqual(u2, 0.0)

    def test_nash_stability(self):
        # Stable: {0, 1}, {2}
        p_stable = Partition(self.game, [{0, 1}, {2}])
        analyzer = StabilityAnalyzer(self.game)
        self.assertTrue(analyzer.is_nash_stable(p_stable))

        # Unstable: {0}, {1}, {2}
        p_unstable = Partition(self.game, [{0}, {1}, {2}])
        self.assertFalse(analyzer.is_nash_stable(p_unstable))

    def test_local_search(self):
        # Start from unstable partition
        p_initial = Partition(self.game, [{0}, {1}, {2}])
        search = SearchAlgorithm(self.game)
        p_final = search.improve_partition(p_initial)
        
        analyzer = StabilityAnalyzer(self.game)
        self.assertTrue(analyzer.is_nash_stable(p_final))
        # Ensure {0, 1} are together
        self.assertEqual(p_final.player_to_coalition[0], p_final.player_to_coalition[1])

if __name__ == '__main__':
    unittest.main()
