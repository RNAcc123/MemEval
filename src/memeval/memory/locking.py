"""Cross-platform advisory file locking for serializing external CLI calls."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Iterator

try:  # POSIX
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # Windows
    fcntl = None
    _HAVE_FCNTL = False

try:  # Windows
    import msvcrt

    _HAVE_MSVCRT = True
except ImportError:  # POSIX
    msvcrt = None
    _HAVE_MSVCRT = False


_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()

_POLL_SECONDS = 0.05


def thread_lock_for_path(path: Path) -> threading.Lock:
    """Return a process-wide thread lock keyed by lock file path."""
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[path] = lock
        return lock


def _acquire_posix(handle, timeout: float | None) -> None:
    if timeout is None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring lock after {timeout}s")
            time.sleep(_POLL_SECONDS)


def _acquire_windows(handle, timeout: float | None) -> None:
    # msvcrt locks byte ranges, so lock a single byte at offset 0 as the sentinel.
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring lock after {timeout}s")
            time.sleep(_POLL_SECONDS)


def _release_windows(handle) -> None:
    with contextlib.suppress(OSError):
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def interprocess_lock(path: Path, *, timeout: float | None = None) -> Iterator[None]:
    """Serialize a critical section across both threads and processes.

    The thread lock is required because POSIX ``flock`` is per open-file-description
    and would not exclude sibling threads sharing this process.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with thread_lock_for_path(path):
        # Binary mode keeps msvcrt byte-range locking well-defined.
        with open(path, "a+b") as handle:
            if _HAVE_FCNTL:
                _acquire_posix(handle, timeout)
            elif _HAVE_MSVCRT:
                _acquire_windows(handle, timeout)
            else:
                raise RuntimeError("No supported file locking primitive on this platform")
            try:
                yield
            finally:
                if _HAVE_FCNTL:
                    with contextlib.suppress(OSError):
                        fcntl.flock(handle, fcntl.LOCK_UN)
                else:
                    _release_windows(handle)


def default_lock_root() -> Path:
    """Lock directory; overridable so parallel workers can be isolated in tests."""
    configured = os.getenv("MEMEVAL_OPENCLAW_SESSION_LOCK_ROOT")
    if configured:
        return Path(configured)
    return Path(os.getenv("TMPDIR") or os.getenv("TEMP") or "/tmp")
