"""Debounce cell crossings into region visits (ADR-0009).

A piece's crossings are mapped to regions, consecutive same-region cells are
merged into candidate visits, and candidates that miss the threshold are
folded into the previous kept visit. A long unassigned gap cuts the sequence
instead of becoming a visit. Spark-free so the regions scan and the
assign-regions UDF share one implementation.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .config import DebounceParameters

# Crossing columns match track_cells: cell_x, cell_y, length_m, entry_x, entry_y, exit_x, exit_y.
Crossing = tuple[int, int, float, float, float, float, float]
_END_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Visit:
    region_id: int
    length_m: float
    entry_x: float
    entry_y: float
    exit_x: float
    exit_y: float
    gap_before: bool


@dataclass(frozen=True, slots=True)
class DebouncedPiece:
    visits: tuple[Visit, ...]
    candidate_visits: int
    kept_visits: int
    cuts: int


@dataclass(slots=True)
class _Candidate:
    region_id: int | None
    length_m: float
    entry: tuple[float, float]
    exit: tuple[float, float]
    start: float
    end: float
    gap_before: bool = False


def debounce_visits(
    crossings: Sequence[Crossing],
    assignment: Mapping[tuple[int, int], int],
    match_offsets_m: Sequence[float],
    parameters: DebounceParameters,
) -> DebouncedPiece:
    """Debounce one piece. `assignment` omits cells that belong to no region."""
    candidates = _candidate_visits(crossings, assignment)
    kept, cuts = _debounce_candidates(candidates, match_offsets_m, parameters)
    visits = tuple(_to_visit(item) for item in kept)
    return DebouncedPiece(
        visits=visits,
        candidate_visits=len(candidates),
        kept_visits=len(visits),
        cuts=cuts,
    )


def _candidate_visits(
    crossings: Sequence[Crossing],
    assignment: Mapping[tuple[int, int], int],
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    travelled = 0.0
    for cell_x, cell_y, length_m, entry_x, entry_y, exit_x, exit_y in crossings:
        region_id = assignment.get((cell_x, cell_y))
        start, travelled = travelled, travelled + length_m
        if candidates and candidates[-1].region_id == region_id:
            candidates[-1].length_m += length_m
            candidates[-1].exit = (exit_x, exit_y)
            candidates[-1].end = travelled
        else:
            candidates.append(
                _Candidate(
                    region_id,
                    length_m,
                    (entry_x, entry_y),
                    (exit_x, exit_y),
                    start,
                    travelled,
                )
            )
    return candidates


def _debounce_candidates(
    candidates: list[_Candidate],
    match_offsets_m: Sequence[float],
    parameters: DebounceParameters,
) -> tuple[list[_Candidate], int]:
    if not candidates:
        return [], 0
    total = candidates[-1].end
    offsets = sorted(match_offsets_m)
    cuts = 0
    changed = True
    while changed:
        changed = False
        kept: list[_Candidate] = []
        pending_cut = False
        for position, item in enumerate(candidates):
            if _is_cut(item, total, offsets, parameters):
                pending_cut = True
                changed = True
                continue
            if not _qualifies(item, total, offsets, parameters):
                if kept and not pending_cut:
                    _merge_into(kept[-1], item)
                    changed = True
                    continue
                if position + 1 < len(candidates) or pending_cut:
                    changed = True
                    continue
                continue
            if (
                kept
                and not pending_cut
                and not item.gap_before
                and kept[-1].region_id == item.region_id
            ):
                _merge_into(kept[-1], item)
                changed = True
            else:
                if pending_cut:
                    item.gap_before = True
                    cuts += 1
                    pending_cut = False
                kept.append(item)
        if pending_cut and kept:
            cuts += 1
        candidates = kept
    return candidates, cuts


def _qualifies(
    item: _Candidate,
    total: float,
    offsets: Sequence[float],
    parameters: DebounceParameters,
) -> bool:
    return item.region_id is not None and _meets_threshold(
        item, total, offsets, parameters
    )


def _is_cut(
    item: _Candidate,
    total: float,
    offsets: Sequence[float],
    parameters: DebounceParameters,
) -> bool:
    return item.region_id is None and _meets_threshold(
        item, total, offsets, parameters
    )


def _meets_threshold(
    item: _Candidate,
    total: float,
    offsets: Sequence[float],
    parameters: DebounceParameters,
) -> bool:
    if item.length_m >= parameters.min_length_m:
        return True
    first = bisect_left(offsets, item.start)
    last = (
        len(offsets)
        if item.end >= total - _END_EPS
        else bisect_left(offsets, item.end)
    )
    return last - first >= parameters.min_match_points


def _merge_into(kept: _Candidate, item: _Candidate) -> None:
    kept.length_m += item.length_m
    kept.exit = item.exit
    kept.end = item.end


def _to_visit(item: _Candidate) -> Visit:
    region_id = item.region_id
    assert region_id is not None
    entry_x, entry_y = item.entry
    exit_x, exit_y = item.exit
    return Visit(
        region_id=region_id,
        length_m=item.length_m,
        entry_x=entry_x,
        entry_y=entry_y,
        exit_x=exit_x,
        exit_y=exit_y,
        gap_before=item.gap_before,
    )
