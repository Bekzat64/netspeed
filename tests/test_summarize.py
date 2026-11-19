"""Unit tests for summarize(): pure function, no network."""

import pytest

from netspeed import RequestResult, summarize


def ok(index: int, size_bytes: int, duration_s: float, ttfb_s: float = 0.1) -> RequestResult:
    return RequestResult(
        index, ok=True, status=200, size_bytes=size_bytes, duration_s=duration_s, ttfb_s=ttfb_s
    )


def test_totals_and_time_stats_ignore_failed_requests() -> None:
    results = [
        ok(1, 1_000_000, 1.0, ttfb_s=0.1),
        ok(2, 2_000_000, 2.0, ttfb_s=0.2),
        ok(3, 3_000_000, 6.0, ttfb_s=0.3),
        # Failed results must be excluded even if they carry numbers.
        RequestResult(4, ok=False, status=500, size_bytes=999, duration_s=9.0, error="HTTP 500"),
    ]

    summary = summarize(results)

    assert summary is not None
    assert summary.succeeded == 3
    assert summary.total_bytes == 6_000_000
    assert summary.mean_s == pytest.approx(3.0)
    assert summary.min_s == 1.0
    assert summary.median_s == 2.0
    assert summary.max_s == 6.0
    assert summary.mean_ttfb_s == pytest.approx(0.2)


def test_speed_is_total_bytes_over_total_time() -> None:
    # Per-request speeds are 80 and 20 Mbit/s; their mean (50) is NOT the throughput.
    results = [ok(1, 10_000_000, 1.0), ok(2, 10_000_000, 4.0)]

    summary = summarize(results)

    assert summary is not None
    # 20 MB in 5 s = 4 MB/s = 32 Mbit/s
    assert summary.speed_mb_s == pytest.approx(4.0)
    assert summary.speed_mbit_s == pytest.approx(32.0)


def test_units_are_decimal_si() -> None:
    # 1,000,000 bytes in 1 s is exactly 1 MB/s and 8 Mbit/s (not 1024-based).
    summary = summarize([ok(1, 1_000_000, 1.0)])

    assert summary is not None
    assert summary.speed_mb_s == pytest.approx(1.0)
    assert summary.speed_mbit_s == pytest.approx(8.0)


@pytest.mark.parametrize(
    "results",
    [
        [RequestResult(1, ok=False, error="timeout"), RequestResult(2, ok=False, error="HTTP 404")],
        [],  # Ctrl+C before the first request completed
    ],
)
def test_no_successful_requests_returns_none(results: list[RequestResult]) -> None:
    assert summarize(results) is None
