"""Server health tracking and recovery for long batch runs.

Workers route every request through a :class:`ServerPool`.  When a request fails with a
transport error the worker asks the pool whether the server is still healthy:

- healthy: the failure belongs to the request (counts as an attempt for that row);
- unhealthy: one worker recovers the server (restarts it when we own it, otherwise waits for
  it to come back) while the others block; requests that hit the outage are retried without
  spending their attempts.

A stall monitor restarts servers we own when requests are in flight but none has finished for
``stall_timeout`` seconds (a hung engine that still answers /health).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import httpx

from .server import api_root


class ServerDead(RuntimeError):
    """A server could not be recovered; the batch stops (resumable)."""


def server_healthy(base_url: str, timeout: float = 10.0) -> bool:
    root = api_root(base_url)
    for path in ("/health", "/v1/models"):
        try:
            if httpx.get(root + path, timeout=timeout).status_code == 200:
                return True
        except httpx.HTTPError:
            continue
    return False


class ServerPool:
    def __init__(
        self,
        urls: list[str],
        *,
        recover: Callable[[int], None] | None = None,
        max_restarts: int = 3,
        server_wait: float = 900.0,
        stall_timeout: float = 1800.0,
        log: Callable[[str], None] = print,
        healthy: Callable[[str], bool] = server_healthy,
    ) -> None:
        self.urls = list(urls)
        self.recover_fn = recover
        self.max_restarts = max_restarts
        self.server_wait = server_wait
        self.stall_timeout = stall_timeout
        self.log = log
        self.healthy = healthy
        n = len(urls)
        self._locks = [threading.Lock() for _ in range(n)]
        self._up = [threading.Event() for _ in range(n)]
        for e in self._up:
            e.set()
        self._gen = [0] * n
        self._state = threading.Lock()
        self.restarts = 0
        self.dead: str | None = None
        self._in_flight = 0
        self._last_progress = time.time()
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None

    # ---------------------------------------------------------- worker side
    def acquire(self, i: int) -> int:
        """Wait until server ``i`` is usable; returns its generation (restart counter)."""
        self._up[i].wait()
        if self.dead:
            raise ServerDead(self.dead)
        with self._state:
            self._in_flight += 1
        return self._gen[i]

    def release(self, *, progressed: bool) -> None:
        with self._state:
            self._in_flight -= 1
            if progressed:
                self._last_progress = time.time()

    def on_transport_error(self, i: int, gen_seen: int) -> bool:
        """True: the server was down (now recovered) -> retry for free.  False: the server is
        healthy, so the error belongs to the request.  Raises ServerDead if unrecoverable."""
        with self._locks[i]:
            if self.dead:
                raise ServerDead(self.dead)
            if self._gen[i] != gen_seen:
                return True  # recovered by another worker after this request started
            if self.healthy(self.urls[i]):
                return False
            self._up[i].clear()
            try:
                self._recover(i, reason="health check failed")
            finally:
                self._up[i].set()
            return True

    # ---------------------------------------------------------- recovery
    def _recover(self, i: int, *, reason: str) -> None:
        url = self.urls[i]
        if self.recover_fn is None:
            self.log(f"[watchdog] {url} unhealthy ({reason}); waiting up to {self.server_wait:.0f}s for it")
            deadline = time.time() + self.server_wait
            while time.time() < deadline:
                if self.healthy(url):
                    self.log(f"[watchdog] {url} is back")
                    self._gen[i] += 1
                    return
                time.sleep(10)
            self.dead = f"{url} did not come back within {self.server_wait:.0f}s"
            raise ServerDead(self.dead)
        with self._state:
            if self.restarts >= self.max_restarts:
                self.dead = f"{url} unhealthy ({reason}) after {self.restarts} restarts (max {self.max_restarts})"
                raise ServerDead(self.dead)
            self.restarts += 1
        self.log(f"[watchdog] restarting {url} ({reason}; restart {self.restarts}/{self.max_restarts})")
        try:
            self.recover_fn(i)
        except Exception as e:
            self.dead = f"restart of {url} failed: {e}"
            raise ServerDead(self.dead) from e
        self._gen[i] += 1
        with self._state:
            self._last_progress = time.time()
        self.log(f"[watchdog] {url} restarted")

    # ---------------------------------------------------------- stall monitor
    def start_monitor(self, interval: float = 30.0) -> None:
        if self.recover_fn is None or self.stall_timeout <= 0:
            return
        self._monitor = threading.Thread(target=self._watch, args=(interval,), daemon=True)
        self._monitor.start()

    def stop_monitor(self) -> None:
        self._stop.set()

    def _watch(self, interval: float) -> None:
        while not self._stop.wait(interval):
            with self._state:
                stalled = self._in_flight > 0 and time.time() - self._last_progress > self.stall_timeout
            if not stalled or self.dead:
                continue
            self.log(f"[watchdog] no request finished for {self.stall_timeout:.0f}s with requests in flight")
            for i in range(len(self.urls)):
                with self._locks[i]:
                    self._up[i].clear()
                    try:
                        self._recover(i, reason="stalled")
                    except ServerDead:
                        return
                    finally:
                        self._up[i].set()
