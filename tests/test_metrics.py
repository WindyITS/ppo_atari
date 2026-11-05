import unittest

from pong_rl.metrics import Episode, EpisodeWindow, summarize_episodes


class ReportingTests(unittest.TestCase):
    def test_positive_timeout_is_not_a_win_and_termination_takes_precedence(self):
        episodes = [Episode(10, 7, 5, False, True), Episode(20, 3, 5, True, True),
                    Episode(30, -4, 5, True, False), Episode(40, 0, 5, True, False)]
        result = summarize_episodes(episodes)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["draws"], 1)
        self.assertEqual(result["timeouts"], 1)
        self.assertEqual(result["positive_timeouts"], 1)
        self.assertEqual(result["completed_win_rate"], 1 / 3)
        self.assertEqual(result["positive_return_fraction"], 0.5)
        self.assertEqual(result["timeout_fraction"], 0.25)

    def test_no_completed_games_has_no_defined_win_rate(self):
        result = summarize_episodes([Episode(5, 10, 5, False, True)])
        self.assertIsNone(result["completed_win_rate"])
        self.assertEqual(result["positive_return_fraction"], 1)

    def test_fixed_step_bounds_include_all_simultaneous_endings(self):
        window = EpisodeWindow(100)
        for step, score in [(50, -21), (100, -10), (101, 5), (200, 10), (200, 15)]:
            window.add(Episode(step, score, 5, True, False))
        result = window.metrics(200)
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["mean_return"], 10)
        self.assertEqual(result["start_step_exclusive"], 100)
        self.assertEqual(result["end_step_inclusive"], 200)

    def test_window_warmup_then_no_recent_endings(self):
        window = EpisodeWindow(100)
        window.add(Episode(10, 8, 5, True, False))
        result = window.metrics(20)
        self.assertFalse(result["full_window"])
        self.assertEqual(result["observed_window_steps"], 20)
        result = window.metrics(110)
        self.assertEqual(result["episodes"], 0)
        self.assertIsNone(result["mean_return"])
        self.assertIsNone(result["completed_win_rate"])
        self.assertTrue(result["full_window"])

    def test_time_cannot_move_back_after_eviction(self):
        window = EpisodeWindow(10)
        window.add(Episode(5, 1, 5, True, False))
        window.metrics(20)
        with self.assertRaises(ValueError):
            window.metrics(15)
        with self.assertRaises(ValueError):
            window.add(Episode(19, 1, 5, True, False))

    def test_window_never_reaches_before_a_finetune_start(self):
        window = EpisodeWindow(100, start_step=1_000)
        window.add(Episode(1_040, 1, 40, True, False))
        early = window.metrics(1_050)
        self.assertEqual((early["start_step_exclusive"], early["observed_window_steps"]), (1_000, 50))
        self.assertFalse(early["full_window"])
        late = window.metrics(1_100)
        self.assertTrue(late["full_window"])
        self.assertEqual(late["episodes"], 1)
        with self.assertRaises(ValueError):
            EpisodeWindow(100, start_step=1_000).metrics(999)


if __name__ == "__main__":
    unittest.main()
