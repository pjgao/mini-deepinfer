"""Physical KV pages + immutable full-prefix sharing.

Each admitted request reserves its worst-case capacity. This intentionally avoids
preemption and deadlock, at the cost of reserving space for ungenerated tokens.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch

from .config import ModelConfig


@dataclass
class Lease:
    pages: list[int]
    cached_tokens: int = 0
    released: bool = False


class PagePool:
    def __init__(self, num_pages: int, block_size: int, prefix_cache: bool = True) -> None:
        if num_pages <= 0 or block_size <= 0:
            raise ValueError("num_pages and block_size must be positive")
        self.num_pages = num_pages
        self.block_size = block_size
        self.prefix_cache = prefix_cache
        self.refs = [0] * num_pages
        self.free = list(reversed(range(num_pages)))
        # Exact token tuples avoid hash collisions. Each entry owns one reference.
        self.prefixes: OrderedDict[tuple[int, ...], int] = OrderedDict()

    def _release_page(self, page: int) -> None:
        assert self.refs[page] > 0
        self.refs[page] -= 1
        if self.refs[page] == 0:
            self.free.append(page)

    def acquire(self, prompt: list[int], capacity: int) -> Lease | None:
        if not prompt or capacity < len(prompt):
            raise ValueError("Capacity must cover a nonempty prompt")
        required = (capacity + self.block_size - 1) // self.block_size
        if required > self.num_pages:
            raise ValueError(f"Request needs {required} pages, pool has {self.num_pages}")
        hits = []
        # Keep at least the last prompt token to recompute next-token logits.
        if self.prefix_cache:
            for end in range(self.block_size, len(prompt), self.block_size):
                key = tuple(prompt[:end])
                if key not in self.prefixes:
                    break
                hits.append(self.prefixes[key])
                self.prefixes.move_to_end(key)
        # Temporarily pin hits so eviction cannot reclaim our prospective prefix.
        for page in hits:
            self.refs[page] += 1
        missing = required - len(hits)
        for key, page in list(self.prefixes.items()):
            if len(self.free) >= missing:
                break
            if self.refs[page] == 1:
                del self.prefixes[key]
                self._release_page(page)
        if len(self.free) < missing:
            for page in hits:
                self._release_page(page)
            return None
        pages = hits.copy()
        for _ in range(missing):
            page = self.free.pop()
            self.refs[page] = 1
            pages.append(page)
        return Lease(pages, len(hits) * self.block_size)

    def remember(self, lease: Lease, prompt: list[int], computed: int) -> None:
        if not self.prefix_cache:
            return
        assert not lease.released
        for end in range(self.block_size, min(computed, len(prompt)) + 1, self.block_size):
            key = tuple(prompt[:end])
            if key not in self.prefixes:
                page = lease.pages[end // self.block_size - 1]
                self.prefixes[key] = page
                self.refs[page] += 1

    def release(self, lease: Lease) -> None:
        if lease.released:
            raise ValueError("Lease already released")
        for page in lease.pages:
            self._release_page(page)
        lease.released = True

    def clear_prefixes(self) -> None:
        for page in self.prefixes.values():
            self._release_page(page)
        self.prefixes.clear()

    def check_invariants(self) -> None:
        assert len(self.free) == len(set(self.free))
        assert set(self.free) == {i for i, count in enumerate(self.refs) if count == 0}
        assert all(count >= 0 for count in self.refs)
        assert all(self.refs[page] > 0 for page in self.prefixes.values())


class PagedKV:
    def __init__(
        self,
        config: ModelConfig,
        num_pages: int,
        block_size: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.block_size = block_size
        self.data = torch.zeros(
            config.num_layers, num_pages, block_size, config.cache_width, device=device, dtype=dtype
        )

    def write(self, layer: int, pages: list[int], start: int, values: torch.Tensor) -> None:
        positions = torch.arange(start, start + len(values), device=self.data.device)
        table = torch.tensor(pages, device=self.data.device, dtype=torch.long)
        self.data[layer, table[positions // self.block_size], positions % self.block_size] = values

    def read(self, layer: int, pages: list[int], length: int) -> torch.Tensor:
        used = pages[: (length + self.block_size - 1) // self.block_size]
        table = torch.tensor(used, device=self.data.device, dtype=torch.long)
        return self.data[layer, table].flatten(0, 1)[:length]

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()
