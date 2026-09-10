import json
import os
import unittest

import pandas as pd

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FEATURES = ["skill_overlap", "n_res_skills", "n_jd_skills", "matched", "missing",
            "missing_ratio", "skill_idf_overlap", "exp_score_norm", "exp_gap_norm",
            "exp_meets", "edu_match", "edu_diff", "log_len_res", "log_len_jd",
            "len_ratio", "same_track", "tfidf_cosine", "sbert_cosine",
            "emb_diff_mean", "emb_diff_std"]


def check_version(v):
    df = pd.read_parquet(os.path.join(PROC, f"features_v{v}.parquet"))
    pairs = pd.read_parquet(os.path.join(PROC, f"pairs_v{v}.parquet"))
    return df, pairs


class FeaturesV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.pairs = check_version("1")

    def test_row_counts_match_pairs(self):
        self.assertEqual(len(self.df), len(self.pairs))
        self.assertEqual(set(self.df.pair_id), set(self.pairs.pair_id))

    def test_no_nan_features(self):
        self.assertEqual(int(self.df[FEATURES].isna().sum().sum()), 0)

    def test_cosine_ranges(self):
        for c in ["tfidf_cosine", "sbert_cosine"]:
            self.assertGreaterEqual(self.df[c].min(), 0)
            self.assertLessEqual(self.df[c].max(), 100)

    def test_sbert_differs_from_proxy(self):
        # Real SBERT must add information beyond the synthetic proxy.
        corr = self.df[["sbert_cosine", "bert_proxy"]].corr().iloc[0, 1]
        self.assertLess(corr, 0.95)
        self.assertGreater(corr, 0.1)

    def test_splits_preserved(self):
        self.assertEqual(set(self.df.split.unique()), {"train", "val", "test"})


class FeaturesV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.pairs = check_version("2")

    def test_row_counts_match_pairs(self):
        self.assertEqual(len(self.df), len(self.pairs))

    def test_no_nan_features(self):
        self.assertEqual(int(self.df[FEATURES].isna().sum().sum()), 0)

    def test_artifacts_exist(self):
        for f in ["tfidf_v2.pkl", "feature_config_v2.json", "embeddings_v2.npz",
                  "feature_stats_v2.json", "features_v2.parquet"]:
            self.assertTrue(os.path.exists(os.path.join(PROC, f)), f)

    def test_config_train_only(self):
        cfg = json.load(open(os.path.join(PROC, "feature_config_v2.json")))
        self.assertTrue(cfg["sbert_available"])
        self.assertEqual(cfg["tfidf"]["fit_on"], "train_split_only")


if __name__ == "__main__":
    unittest.main()
