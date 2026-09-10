import os
import unittest

import pandas as pd

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class DatasetV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_parquet(os.path.join(PROC, "pairs_v1.parquet"))

    def test_files_exist(self):
        for f in ["pairs_v1.parquet", "train_v1.parquet", "val_v1.parquet", "test_v1.parquet", "stats_v1.json"]:
            self.assertTrue(os.path.exists(os.path.join(PROC, f)), f)

    def test_score_ranges(self):
        for col in ["skill_score", "experience_score", "education_score", "bert_proxy", "ats_proxy"]:
            self.assertGreaterEqual(self.df[col].min(), 0)
            self.assertLessEqual(self.df[col].max(), 100)

    def test_no_resume_leakage(self):
        tr = set(self.df[self.df.split == "train"].resume_id)
        va = set(self.df[self.df.split == "val"].resume_id)
        te = set(self.df[self.df.split == "test"].resume_id)
        self.assertFalse(tr & va)
        self.assertFalse(tr & te)
        self.assertFalse(va & te)

    def test_label_balance(self):
        rate = self.df.label_match.mean()
        self.assertGreaterEqual(rate, 0.25)
        self.assertLessEqual(rate, 0.75)

    def test_splits_cover_all(self):
        self.assertEqual(set(self.df.split.unique()), {"train", "val", "test"})


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_parquet(os.path.join(PROC, "pairs_v2.parquet"))

    def test_files_exist(self):
        for f in ["pairs_v2.parquet", "train_v2.parquet", "val_v2.parquet", "test_v2.parquet",
                  "stats_v2.json", "../gold/gold_sample_v2.csv", "../raw/kaggle/kaggle_resume.parquet"]:
            self.assertTrue(os.path.exists(os.path.join(PROC, f)), f)

    def test_provenance_columns(self):
        self.assertIn("source", self.df.columns)
        self.assertIn("kaggle_category", self.df.columns)
        self.assertEqual(set(self.df.source.unique()), {"synthetic", "kaggle"})

    def test_no_resume_leakage(self):
        tr = set(self.df[self.df.split == "train"].resume_id)
        va = set(self.df[self.df.split == "val"].resume_id)
        te = set(self.df[self.df.split == "test"].resume_id)
        self.assertFalse(tr & va)
        self.assertFalse(tr & te)
        self.assertFalse(va & te)

    def test_sources_disjoint_resumes(self):
        syn = set(self.df[self.df.source == "synthetic"].resume_id)
        kag = set(self.df[self.df.source == "kaggle"].resume_id)
        self.assertFalse(syn & kag)

    def test_has_positives_from_synthetic(self):
        # kaggle slice is ~99% negatives (non-tech resumes vs tech JDs);
        # synthetic slice must supply the positives for training.
        syn_rate = self.df[self.df.source == "synthetic"].label_match.mean()
        self.assertGreaterEqual(syn_rate, 0.25)
        self.assertGreaterEqual(self.df.label_match.sum(), 1000)

    def test_score_ranges(self):
        for col in ["skill_score", "experience_score", "education_score", "bert_proxy", "ats_proxy"]:
            self.assertGreaterEqual(self.df[col].min(), 0)
            self.assertLessEqual(self.df[col].max(), 100)


if __name__ == "__main__":
    unittest.main()
