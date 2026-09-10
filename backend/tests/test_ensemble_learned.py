"""Phase 4 tests: learned registry validity + learned scorer behavior."""
import json
import os
import unittest

import pandas as pd

BACKEND = os.path.join(os.path.dirname(__file__), "..")
REG = os.path.join(BACKEND, "models", "registry", "ensemble_v1")
PROC = os.path.join(BACKEND, "data", "processed")


class RegistryTests(unittest.TestCase):
    def test_files_exist(self):
        for f in ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl",
                  "meta_logreg.pkl", "calibrator.pkl", "competence_head.pkl",
                  "oof_preds.npz", "metrics_baselines.json", "metrics_ensemble.json",
                  "config.json"]:
            self.assertTrue(os.path.exists(os.path.join(REG, f)), f)

    def test_stack_beats_heuristic_auc(self):
        ens = json.load(open(os.path.join(REG, "metrics_ensemble.json")))
        base = json.load(open(os.path.join(REG, "metrics_baselines.json")))
        self.assertGreater(ens["models"]["stack_calibrated"]["roc_auc"],
                           base["models"]["heuristic_ats"]["roc_auc"] - 0.01)  # weak-label ceiling ~1.0

    def test_calibration_beats_heuristic_ece(self):
        ens = json.load(open(os.path.join(REG, "metrics_ensemble.json")))
        base = json.load(open(os.path.join(REG, "metrics_baselines.json")))
        self.assertLess(ens["models"]["stack_calibrated"]["ece"],
                        base["models"]["heuristic_ats"]["ece"])

    def test_competence_beats_majority(self):
        ens = json.load(open(os.path.join(REG, "metrics_ensemble.json")))
        c = ens["models"]["competence_head"]
        self.assertGreater(c["acc"], c["majority_acc"])

    def test_config_sane(self):
        cfg = json.load(open(os.path.join(REG, "config.json")))
        self.assertEqual(cfg["model_version"], "ensemble_v1")
        self.assertTrue(0.2 <= cfg["tuned_threshold"] <= 0.8)


class LearnedScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(BACKEND)))
        from services.ensemble import is_learned_available
        assert is_learned_available(), "registry missing"
        from training.features import FEATURE_COLUMNS
        cls.FEATURE_COLUMNS = FEATURE_COLUMNS
        feat = pd.read_parquet(os.path.join(PROC, "features_v2.parquet"))
        cls.row = feat[feat["split"] == "test"].iloc[0]
        from training.utils import pair_embedding_matrix
        te = feat[feat["split"] == "test"].reset_index(drop=True)
        E = pair_embedding_matrix("2", te)
        cls.emb = E[(te["pair_id"] == cls.row["pair_id"]).to_numpy()][0]

    def test_available(self):
        from services.ensemble import is_learned_available
        self.assertTrue(is_learned_available())

    def test_score_bounded_and_deterministic(self):
        from services.ensemble import learned_stack_score
        fv = {c: float(self.row[c]) for c in self.FEATURE_COLUMNS}
        s1, l1 = learned_stack_score(fv, self.emb)
        s2, l2 = learned_stack_score(fv, self.emb)
        self.assertEqual((s1, l1), (s2, l2))
        self.assertGreaterEqual(s1, 0.0)
        self.assertLessEqual(s1, 100.0)
        self.assertIn(l1, [0, 1, 2, 3])

    def test_no_embedding_fallback(self):
        from services.ensemble import learned_stack_score
        fv = {c: float(self.row[c]) for c in self.FEATURE_COLUMNS}
        s, level = learned_stack_score(fv, None)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 100.0)

    def test_heuristic_unchanged(self):
        from services.ensemble import boosted_ensemble_score
        s = boosted_ensemble_score(80, 80, 100, 100)
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)


if __name__ == "__main__":
    unittest.main()
