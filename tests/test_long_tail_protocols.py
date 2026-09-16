from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.build_long_tail_protocols import (
    add_hmt_groups,
    add_log_quantiles,
    allocate_integer_quotas,
    effective_number,
    p11_profiles,
    p12_quotas,
)


class LongTailProtocolConstructionTest(unittest.TestCase):
    def test_log_quantiles_are_head_to_tail_and_cover_every_class(self) -> None:
        frame = pd.DataFrame({"class_id": range(12), "count": np.arange(1, 13)})
        result = add_log_quantiles(frame, 4)
        groups = result.groupby("profile_group")["count"]
        self.assertEqual(groups.size().to_dict(), {"Q1": 3, "Q2": 3, "Q3": 3, "Q4": 3})
        self.assertGreater(groups.min()["Q1"], groups.max()["Q4"])

    def test_effective_number_is_stable_and_monotonic(self) -> None:
        values = effective_number(np.asarray([1, 10, 100_000]), 0.9999)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertAlmostEqual(values[0], 1.0)

    def test_integer_quota_apportionment_preserves_both_margins(self) -> None:
        sizes = [4, 3, 3]
        counts = {"head": 2, "medium": 3, "tail": 5}
        profile = {"head": 0.2, "medium": 0.3, "tail": 0.5}
        quotas = allocate_integer_quotas(sizes, counts, [profile] * 3, seed=7)
        self.assertEqual([sum(row.values()) for row in quotas], sizes)
        self.assertEqual(
            {group: sum(row[group] for row in quotas) for group in counts}, counts
        )

    def test_p11_target_profiles_have_opposing_head_tail_drift(self) -> None:
        args = SimpleNamespace(
            p11_head_max=0.40,
            p11_head_min=0.05,
            p11_tail_min=0.25,
            p11_tail_max=0.65,
        )
        profiles = p11_profiles(8, args)
        self.assertGreater(profiles[0]["head"], profiles[-1]["head"])
        self.assertLess(profiles[0]["tail"], profiles[-1]["tail"])
        for profile in profiles:
            self.assertAlmostEqual(sum(profile.values()), 1.0)

    def test_p12_default_shock_is_exact_and_remaining_profile_is_feasible(self) -> None:
        sizes = [38] + [20] * 7
        group_counts = {"head": 39, "medium": 55, "tail": 84}
        quotas, profiles = p12_quotas(
            sizes,
            group_counts,
            shock_task=3,
            shock_profile=[0.1, 0.1, 0.8],
            seed=0,
        )
        self.assertEqual(quotas[3], {"head": 2, "medium": 2, "tail": 16})
        self.assertEqual([sum(row.values()) for row in quotas], sizes)
        self.assertEqual(
            {group: sum(row[group] for row in quotas) for group in group_counts},
            group_counts,
        )
        self.assertEqual(profiles[3], {"head": 0.1, "medium": 0.1, "tail": 0.8})

    def test_hmt_thresholds_match_study_definition(self) -> None:
        frame = pd.DataFrame({"class_id": [1, 2, 3], "count": [100, 101, 1001]})
        self.assertEqual(
            add_hmt_groups(frame)["profile_group"].tolist(),
            ["tail", "medium", "head"],
        )


if __name__ == "__main__":
    unittest.main()
