import pytest
from src.data.trap_dataset import compute_trap_priority, THEME_WEIGHTS, THEMES


class TestTrapPriority:
    def test_priority_mate_threat_max(self):
        assert compute_trap_priority(1.0, "mate_threat") == 1.0

    def test_priority_fork_mid(self):
        assert compute_trap_priority(0.5, "fork") == 0.5 * 0.7

    def test_priority_other_low(self):
        assert compute_trap_priority(0.1, "other") == 0.1 * 0.3

    def test_all_themes_have_weights(self):
        for theme in THEMES:
            assert theme in THEME_WEIGHTS

    def test_zero_improvement(self):
        assert compute_trap_priority(0.0, "sacrifice") == 0.0
