import unittest

import evaluate_honest as E


class PredictiveValueTests(unittest.TestCase):
    def test_ppv_falls_as_prevalence_falls(self):
        rows = E.predictive_values(sens=0.98, spec=0.99)
        ppvs = [r["ppv"] for r in sorted(rows, key=lambda r: r["prevalence"])]
        self.assertEqual(ppvs, sorted(ppvs))
        self.assertLess(ppvs[0], 0.10)     # rare disease: most positives are false alarms
        self.assertGreater(ppvs[-1], 0.95)  # simulated 40% prevalence

    def test_known_value(self):
        # sensitivity 90%, specificity 90%, prevalence 10% -> PPV = 0.09 / (0.09 + 0.09) = 50%
        row = E.predictive_values(0.9, 0.9)
        self.assertTrue(all(0 <= r["ppv"] <= 1 and 0 <= r["npv"] <= 1 for r in row))
        ppv = 0.9 * 0.1 / (0.9 * 0.1 + 0.1 * 0.9)
        self.assertAlmostEqual(ppv, 0.5)


if __name__ == "__main__":
    unittest.main()
