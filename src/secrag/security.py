"""Access control for the API: optional API key, client IP resolution, rate limiting.

Stdlib only, so the unit tests run in CI without FastAPI or the ML stack installed.
"""

import hmac
import os
import threading
import time
from collections import deque
from collections.abc import Callable


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def api_key_ok(configured: str | None, provided: str | None) -> bool:
    """True when no key is configured (public demo) or the provided key matches."""
    if not configured:
        return True
    if not provided:
        return False
    return hmac.compare_digest(configured.encode(), provided.encode())


def bearer_or_header(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


def client_ip(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """The address of the caller, trusting X-Forwarded-For only for `trusted_hops` proxies.

    Each trusted proxy appends the address it received the request from, so the client is
    the entry `trusted_hops` from the right. Anything left of that was written by the
    client and can be forged, so the left-most entry is never used on its own say-so.
    """
    peer = peer or "unknown"
    if trusted_hops <= 0 or not forwarded_for:
        return peer
    entries = [e.strip() for e in forwarded_for.split(",") if e.strip()]
    if not entries:
        return peer
    # Fewer entries than hops means the chain is shorter than configured; the left-most
    # entry is then the best we have and was still written by our own proxy.
    return entries[-min(trusted_hops, len(entries))]


class RateLimiter:
    """Sliding-window limiter keyed by caller. A limit of 0 disables it.

    In memory: it resets on restart and isn't shared across worker processes, which is
    fine for a single-process demo.
    """

    def __init__(
        self,
        limit: int,
        window_s: float,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = limit
        self.window_s = window_s
        self.max_keys = max_keys
        self.clock = clock
        self.hits: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        q = self.hits.get(key)
        if q is None:
            return deque()
        while q and now - q[0] >= self.window_s:
            q.popleft()
        return q

    def allow(self, key: str = "") -> bool:
        if self.limit <= 0:
            return True
        with self.lock:
            now = self.clock()
            q = self._prune(key, now)
            if len(q) >= self.limit:
                return False
            if key not in self.hits:
                if len(self.hits) >= self.max_keys:
                    # Bound memory under a flood of distinct addresses: drop idle keys first.
                    for k in [k for k in self.hits if not self._prune(k, now)]:
                        del self.hits[k]
                    if len(self.hits) >= self.max_keys:
                        return False
                self.hits[key] = q
            q.append(now)
            return True
