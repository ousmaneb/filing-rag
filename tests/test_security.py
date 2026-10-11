import pytest

from secrag.security import RateLimiter, api_key_ok, bearer_or_header, client_ip, env_int


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_no_configured_key_keeps_the_demo_public():
    assert api_key_ok(None, None)
    assert api_key_ok("", "anything")


def test_configured_key_must_match():
    assert api_key_ok("s3cret", "s3cret")
    assert not api_key_ok("s3cret", None)
    assert not api_key_ok("s3cret", "")
    assert not api_key_ok("s3cret", "s3cre")


def test_key_from_header_or_bearer():
    assert bearer_or_header(None, " k ") == "k"
    assert bearer_or_header("Bearer k", None) == "k"
    assert bearer_or_header("bearer k", None) == "k"
    assert bearer_or_header("Basic k", None) is None
    assert bearer_or_header("Bearer ", None) is None
    assert bearer_or_header(None, None) is None


def test_forwarded_for_ignored_without_trusted_proxies():
    assert client_ip("10.0.0.5", "1.2.3.4", 0) == "10.0.0.5"


def test_forwarded_for_uses_the_entry_our_proxy_wrote():
    # The client forged "6.6.6.6"; the one proxy in front appended the real address.
    assert client_ip("10.0.0.5", "6.6.6.6, 203.0.113.9", 1) == "203.0.113.9"
    assert client_ip("10.0.0.5", "6.6.6.6, 203.0.113.9, 10.0.0.2", 2) == "203.0.113.9"
    assert client_ip("10.0.0.5", "203.0.113.9", 3) == "203.0.113.9"
    assert client_ip("10.0.0.5", None, 1) == "10.0.0.5"
    assert client_ip(None, None, 0) == "unknown"


def test_limiter_is_per_key_and_slides():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, clock=clock)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    clock.now = 59
    assert not limiter.allow("a")
    clock.now = 60
    assert limiter.allow("a")


def test_limit_zero_disables():
    limiter = RateLimiter(0, 60)
    assert all(limiter.allow("a") for _ in range(100))


def test_limiter_memory_is_bounded():
    clock = FakeClock()
    limiter = RateLimiter(1, 60, max_keys=2, clock=clock)
    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("c")
    clock.now = 60
    # Idle keys are evicted to make room.
    assert limiter.allow("c")
    assert len(limiter.hits) <= 2


def test_env_int(monkeypatch):
    monkeypatch.delenv("SECRAG_TEST_INT", raising=False)
    assert env_int("SECRAG_TEST_INT", 7) == 7
    monkeypatch.setenv("SECRAG_TEST_INT", "3")
    assert env_int("SECRAG_TEST_INT", 7) == 3
    monkeypatch.setenv("SECRAG_TEST_INT", "-1")
    with pytest.raises(ValueError):
        env_int("SECRAG_TEST_INT", 7)
