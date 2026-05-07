# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-side handle map for the manual KV-cache pipeline.

The actual KV blobs live on the worker (pinned host memory). This module
holds only the bookkeeping the API process needs:

    fingerprint  ->  ManualKVHandleEntry(handle, prefix_len)

When the LRU evicts a fingerprint, the API process notifies the worker so
the matching pinned-memory blob can be freed.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ManualKVHandleEntry:
    handle: str
    prefix_len: int = 0
    hits: int = 0
    populated: bool = False
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class ManualKVStore:
    """Thread-safe LRU map: fingerprint -> ManualKVHandleEntry.

    Pure bookkeeping. Calls ``on_evict(handle)`` when an entry is
    dropped, so the caller can fire ``manual_kv_drop`` on the worker.
    """

    def __init__(
        self,
        max_entries: int = 64,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, ManualKVHandleEntry] = OrderedDict()
        self.max_entries = max_entries
        self._on_evict = on_evict
        self._lookups = 0
        self._hits = 0

    @staticmethod
    def new_handle() -> str:
        return f"mkv-{uuid.uuid4().hex}"

    # --- read path -----------------------------------------------------
    def get(self, key: str) -> ManualKVHandleEntry | None:
        with self._lock:
            self._lookups += 1
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._hits += 1
            entry.hits += 1
            entry.last_used = time.time()
            self._entries.move_to_end(key)
            return entry

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    # --- write path ----------------------------------------------------
    def reserve(self, key: str) -> ManualKVHandleEntry:
        with self._lock:
            entry = ManualKVHandleEntry(handle=self.new_handle())
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._evict_locked()
            return entry

    def populate(self, key: str, prefix_len: int) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.prefix_len = prefix_len
            entry.populated = True
            entry.last_used = time.time()
            self._entries.move_to_end(key)

    def drop(self, key: str) -> str | None:
        with self._lock:
            entry = self._entries.pop(key, None)
            return entry.handle if entry is not None else None

    # --- introspection -------------------------------------------------
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            populated = sum(1 for e in self._entries.values() if e.populated)
            return {
                "entries": len(self._entries),
                "populated": populated,
                "max_entries": self.max_entries,
                "lookups": self._lookups,
                "hits": self._hits,
                "hit_rate": (self._hits / self._lookups) if self._lookups else 0.0,
            }

    # --- internal ------------------------------------------------------
    def _evict_locked(self) -> None:
        while len(self._entries) > self.max_entries:
            _, victim = self._entries.popitem(last=False)
            if self._on_evict is not None:
                try:
                    self._on_evict(victim.handle)
                except Exception:
                    # Eviction is best-effort; never let it break the
                    # request that triggered it.
                    pass
