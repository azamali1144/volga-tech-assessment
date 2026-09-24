import threading
import unittest

from app.rate_limit import RateLimiter


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RateLimiterTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.limiter = RateLimiter(max_requests=3, window_seconds=60, clock=self.clock)

    def test_allows_up_to_limit_then_rejects(self):
        decisions = [self.limiter.check("k") for _ in range(4)]
        self.assertEqual([d.allowed for d in decisions], [True, True, True, False])
        self.assertEqual([d.remaining for d in decisions], [2, 1, 0, 0])
        self.assertEqual(decisions[-1].limit, 3)

    def test_keys_are_limited_independently(self):
        for _ in range(3):
            self.limiter.check("alice")
        self.assertFalse(self.limiter.check("alice").allowed)
        self.assertTrue(self.limiter.check("bob").allowed)

    def test_retry_after_counts_down_to_oldest_hit_expiring(self):
        self.limiter.check("k")            # t=0
        self.clock.advance(10)
        self.limiter.check("k")            # t=10
        self.limiter.check("k")            # t=10
        self.clock.advance(5)              # t=15: oldest expires at t=60
        rejected = self.limiter.check("k")
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.retry_after_seconds, 45)

    def test_retry_after_is_whole_seconds_and_at_least_one(self):
        for _ in range(3):
            self.limiter.check("k")
        self.clock.advance(59.6)
        self.assertEqual(self.limiter.check("k").retry_after_seconds, 1)  # ceil(0.4)

    def test_window_slides_rather_than_resetting(self):
        self.limiter.check("k")            # t=0
        self.clock.advance(30)
        self.limiter.check("k")            # t=30
        self.limiter.check("k")            # t=30
        self.clock.advance(30)             # t=60: only the t=0 hit has expired
        self.assertTrue(self.limiter.check("k").allowed)
        self.assertFalse(self.limiter.check("k").allowed)
        self.clock.advance(30)             # t=90: the two t=30 hits expire
        self.assertTrue(self.limiter.check("k").allowed)

    def test_no_double_limit_burst_at_window_boundary(self):
        """A fixed window would allow 3 at t=59 and 3 more at t=61."""
        self.clock.advance(59)
        for _ in range(3):
            self.assertTrue(self.limiter.check("k").allowed)
        self.clock.advance(2)
        self.assertFalse(self.limiter.check("k").allowed)

    def test_rejected_requests_do_not_extend_the_block(self):
        for _ in range(3):
            self.limiter.check("k")
        for _ in range(50):                # hammering while limited
            self.clock.advance(1)
            self.limiter.check("k")
        self.clock.advance(10)             # t=60: original hits expired
        self.assertTrue(self.limiter.check("k").allowed)

    def test_idle_keys_are_swept_to_bound_memory(self):
        for i in range(100):
            self.limiter.check(f"key-{i}")
        self.assertEqual(self.limiter.tracked_keys, 100)
        self.clock.advance(61)
        self.limiter.check("active")
        self.assertEqual(self.limiter.tracked_keys, 1)

    def test_thread_safe_under_concurrency(self):
        limiter = RateLimiter(max_requests=50, window_seconds=60, clock=self.clock)
        allowed = []
        lock = threading.Lock()

        def hammer():
            for _ in range(20):
                if limiter.check("shared").allowed:
                    with lock:
                        allowed.append(1)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(allowed), 50)  # exactly the limit, never more

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            RateLimiter(max_requests=0, window_seconds=60)
        with self.assertRaises(ValueError):
            RateLimiter(max_requests=1, window_seconds=0)


if __name__ == "__main__":
    unittest.main()
