# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block ownership index — build step 1 (02 §10).

Maps `(job_id, node, call_type)` to the blocks that node's prefix occupies,
and back. Deliberately knows nothing about Redis or scoring: it is a pure
data structure, maintained from `BlockPool.cache_full_blocks` and
`BlockPool._maybe_evict_cached_block`, and testable against a fake pool.

Two properties carry most of the design:

- **Multi-owner.** A shared system preamble belongs to every node whose
  prefix starts with it, so the reverse map holds a *set* of keys. The
  previous attempt used a single owner per block and produced the `RETAG`
  thrash where the last writer stole a common prefix (00 Part 2, weakness 4).
- **Ordered, with gaps.** Position within the node's prefix is recorded so
  eviction can be tail-first: a prefix is only worth anything as a contiguous
  run from position 0, so freeing from the middle destroys the whole match
  for one block of reclaim. `remove_skipped_blocks` punches holes in the
  middle for windowed and mamba models, so positions must tolerate gaps.
"""

from collections.abc import Iterable, Iterator

from vllm.v1.core.node_eviction.types import NodeKey


class NodeEntry:
    """The blocks currently held by one `(job, node, call_type)` key."""

    __slots__ = (
        "key",
        "positions",
        "speculative",
        "created_at",
        "last_seen",
        "max_position",
    )

    def __init__(self, key: NodeKey, created_at: float) -> None:
        self.key = key
        # block_id -> position within this node's prefix. A dict rather than
        # a list because positions have gaps (see module docstring).
        self.positions: dict[int, int] = {}
        # True while the entry exists only because a phantom prefetch created
        # it, i.e. it was produced by a prediction rather than an observation
        # (02 §5). Cleared on the first *real* prefix hit.
        self.speculative = False
        self.created_at = created_at
        self.last_seen = created_at
        # Highest position ever recorded, used as the run length in the
        # decision log so tail-first conformance is checkable (07 §4).
        self.max_position = -1

    @property
    def num_blocks(self) -> int:
        return len(self.positions)

    @property
    def run_len(self) -> int:
        return self.max_position + 1

    def __repr__(self) -> str:
        return (
            f"NodeEntry(key={self.key}, blocks={self.num_blocks}, "
            f"speculative={self.speculative})"
        )


class BlockOwnershipIndex:
    """`(job, node, call_type) <-> block_id`, with positions and provenance.

    Lives on the scheduler thread and is only touched from it, so there is
    no locking here by construction.
    """

    def __init__(self) -> None:
        self._entries: dict[NodeKey, NodeEntry] = {}
        self._owners: dict[int, set[NodeKey]] = {}

    # -- maintenance hooks -------------------------------------------------

    def add_blocks(
        self,
        key: NodeKey,
        block_ids: Iterable[tuple[int, int]],
        now: float,
        speculative: bool = False,
    ) -> NodeEntry:
        """Record `(block_id, prefix_position)` pairs as owned by `key`.

        Called from `cache_full_blocks`, where blocks arrive in prefix order
        and the position is the enumeration index.
        """
        entry = self._entries.get(key)
        if entry is None:
            entry = NodeEntry(key, now)
            entry.speculative = speculative
            self._entries[key] = entry
        elif not speculative and entry.speculative:
            # A real request has now written blocks under this key, so the
            # entry is no longer a pure prediction.
            entry.speculative = False
        entry.last_seen = now

        positions = entry.positions
        owners = self._owners
        for block_id, position in block_ids:
            positions[block_id] = position
            if position > entry.max_position:
                entry.max_position = position
            owner_set = owners.get(block_id)
            if owner_set is None:
                owners[block_id] = {key}
            else:
                owner_set.add(key)
        return entry

    def remove_block(self, block_id: int) -> None:
        """Drop a block from every key that owns it.

        Called from `_maybe_evict_cached_block`, i.e. at the moment the block
        loses its prefix-cache identity.
        """
        owner_set = self._owners.pop(block_id, None)
        if not owner_set:
            return
        for key in owner_set:
            entry = self._entries.get(key)
            if entry is None:
                continue
            entry.positions.pop(block_id, None)
            if not entry.positions:
                # An entry with no blocks carries no information; dropping it
                # keeps the index proportional to what is actually resident.
                del self._entries[key]

    def touch_key(self, key: NodeKey, now: float) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.last_seen = now

    def confirm(self, key: NodeKey, now: float) -> bool:
        """Flip a speculative entry to confirmed.

        Driven by a real (non-prefetch) prefix hit: that *is* the signal that
        the prediction came true, and it needs no cooperation from the
        forecast side (02 §5 part 2).

        Returns True if this call changed the entry's state.
        """
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry.last_seen = now
        if not entry.speculative:
            return False
        entry.speculative = False
        return True

    def confirm_block(self, block_id: int, now: float) -> int:
        """Confirm every key owning `block_id`. Returns how many flipped."""
        owner_set = self._owners.get(block_id)
        if not owner_set:
            return 0
        return sum(1 for key in tuple(owner_set) if self.confirm(key, now))

    def drop_key(self, key: NodeKey) -> int:
        """Remove one key entirely. Returns the number of blocks released.

        A block co-owned by another key stays in the reverse map; only this
        key's claim on it goes away.
        """
        entry = self._entries.pop(key, None)
        if entry is None:
            return 0
        for block_id in entry.positions:
            owner_set = self._owners.get(block_id)
            if owner_set is None:
                continue
            owner_set.discard(key)
            if not owner_set:
                del self._owners[block_id]
        return len(entry.positions)

    def gc(self, now: float, hard_drop_age: float) -> int:
        """Drop entries the policy can no longer say anything useful about.

        One clock. A silent job decays out of the *score* via the time
        discount, but the index would still hold its keys forever, so there
        is a hard age drop on `last_seen`.

        Speculative entries used to drop on a second clock — `TTL x multiple`
        past creation, on the theory that a prediction that far past its own
        deadline has been falsified. That TTL came from
        `time_to_next_call_ms` and has been removed along with the floor
        decay it shared a clock with (see `scoring.speculative_floor`).
        Keeping it here would have been the same expiry wearing a different
        name: a prefetched prefix held at the floor by the scorer, then
        deleted out from under it by the GC.

        Returns the number of keys dropped.
        """
        doomed = [
            key
            for key, entry in self._entries.items()
            if now - entry.last_seen >= hard_drop_age
        ]
        for key in doomed:
            self.drop_key(key)
        return len(doomed)

    def clear(self) -> None:
        """Drop everything. Driven by `reset_prefix_cache`, which invalidates
        every block hash in the pool."""
        self._entries.clear()
        self._owners.clear()

    # -- reads -------------------------------------------------------------

    def get_entry(self, key: NodeKey) -> NodeEntry | None:
        return self._entries.get(key)

    def owners_of(self, block_id: int) -> set[NodeKey]:
        return self._owners.get(block_id, frozenset())  # type: ignore[return-value]

    def position_of(self, key: NodeKey, block_id: int) -> int | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        return entry.positions.get(block_id)

    def entries(self) -> Iterator[NodeEntry]:
        return iter(self._entries.values())

    def keys(self) -> Iterator[NodeKey]:
        return iter(self._entries.keys())

    def tracked_block_ids(self) -> Iterator[int]:
        return iter(self._owners.keys())

    def tracked_block_owners(self) -> list[tuple[int, set[NodeKey]]]:
        """Every indexed block with its owning keys.

        Materialised: this is the splice's candidate set, and the caller must
        be free to mutate the index while walking it.
        """
        return list(self._owners.items())

    @property
    def num_keys(self) -> int:
        return len(self._entries)

    @property
    def num_blocks(self) -> int:
        return len(self._owners)

    @property
    def num_speculative_keys(self) -> int:
        return sum(1 for e in self._entries.values() if e.speculative)
