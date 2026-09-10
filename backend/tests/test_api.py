"""Phase 6/7 API tests: /analyze legacy+ML fields, validation, /rank ordering."""
import io
import os
import unittest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STRONG = ("Python developer with 5 years of experience. Skills: python, django, "
          "postgresql, docker, aws, git, linux. B.Tech in Computer Science.")
WEAK = ("Chef with 10 years of experience in hospitality and catering. "
        "Skills: cooking, baking, menu planning.")
JD = ("Backend Developer. Requirements: python, django, postgresql, docker. "
      "Minimum 3 years of relevant experience required. "
      "Bachelor's degree in Computer Science required.")


def make_pdf(text):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, BACKEND)
        from fastapi.testclient import TestClient
        from app import app
        cls.client = TestClient(app)

    def test_analyze_happy_path(self):
        r = self.client.post("/analyze",
                             files={"resume": ("r.pdf", make_pdf(STRONG), "application/pdf")},
                             data={"job_description": JD})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        for k in ["ats_score", "ensemble_score", "bert_similarity", "matched_skills",
                  "decision", "ml_score", "ml_confidence", "ml_competence",
                  "model_version", "top_features"]:
            self.assertIn(k, d, k)
        self.assertEqual(d["model_version"], "ensemble_v1")
        self.assertGreaterEqual(d["ml_score"], 0)
        self.assertLessEqual(d["ml_score"], 100)
        self.assertEqual(len(d["top_features"]), 5)
        # legacy math unchanged
        self.assertAlmostEqual(d["ats_score"],
                               round(0.4 * d["skill_score"] + 0.25 * d["bert_similarity"] +
                                     0.2 * d["experience_score"] + 0.15 * d["education_score"], 2))

    def test_analyze_rejects_non_pdf(self):
        r = self.client.post("/analyze",
                             files={"resume": ("r.txt", b"hello", "text/plain")},
                             data={"job_description": JD})
        self.assertEqual(r.status_code, 400)

    def test_rank_orders_strong_first(self):
        r = self.client.post("/rank",
                             files=[("resumes", ("weak.pdf", make_pdf(WEAK), "application/pdf")),
                                    ("resumes", ("strong.pdf", make_pdf(STRONG), "application/pdf"))],
                             data={"job_description": JD})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["count"], 2)
        self.assertEqual(d["results"][0]["filename"], "strong.pdf")
        self.assertEqual([x["rank"] for x in d["results"]], [1, 2])

    def test_rank_limit(self):
        files = [("resumes", (f"r{i}.pdf", make_pdf(STRONG), "application/pdf")) for i in range(11)]
        r = self.client.post("/rank", files=files, data={"job_description": JD})
        self.assertEqual(r.status_code, 400)

    def test_ml_disabled_fallback(self):
        os.environ["USE_ML_MODEL"] = "0"
        try:
            r = self.client.post("/analyze",
                                 files={"resume": ("r.pdf", make_pdf(STRONG), "application/pdf")},
                                 data={"job_description": JD})
            self.assertEqual(r.status_code, 200)
            d = r.json()
            self.assertEqual(d["model_version"], "heuristic")
            self.assertIsNone(d["ml_score"])
            self.assertIn("ats_score", d)
        finally:
            del os.environ["USE_ML_MODEL"]


if __name__ == "__main__":
    unittest.main()
