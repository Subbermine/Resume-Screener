import unittest

from services.ensemble import boosted_ensemble_score


class EnsembleScoreTests(unittest.TestCase):
    def test_boosted_score_is_bounded_and_ranked(self):
        related = boosted_ensemble_score(83.61, 86.0, 100.0, 100.0)
        unrelated = boosted_ensemble_score(10.72, 20.0, 0.0, 0.0)

        self.assertGreaterEqual(related, 0)
        self.assertLessEqual(related, 100)
        self.assertGreaterEqual(unrelated, 0)
        self.assertLessEqual(unrelated, 100)
        self.assertGreater(related, unrelated)


if __name__ == "__main__":
    unittest.main()
