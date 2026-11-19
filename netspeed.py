"""Measure download speed by fetching one URL several times in a row."""

import argparse
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

BYTES_PER_MB = 1_000_000  # decimal SI megabyte
BITS_PER_MBIT = 1_000_000  # decimal SI megabit
CHUNK_SIZE = 64 * 1024
USER_AGENT = "netspeed/1.0"

ROW_FORMAT = "{:>3}  {:>6} {:>10} {:>9} {:>9} {:>14}"
TABLE_HEADER = ROW_FORMAT.format("#", "status", "size", "time", "TTFB", "speed")
SEPARATOR = "-" * len(TABLE_HEADER)


@dataclass(frozen=True)
class RequestResult:
    """Outcome of one request. Sizes and times are meaningful only when ok is True."""

    index: int
    ok: bool
    status: int | None = None
    size_bytes: int = 0
    duration_s: float = 0.0
    ttfb_s: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class Summary:
    """Metrics computed over successful requests only."""

    succeeded: int
    total_bytes: int
    mean_s: float
    min_s: float
    median_s: float
    max_s: float
    mean_ttfb_s: float
    speed_mbit_s: float
    speed_mb_s: float


def measure_once(client: httpx.Client, url: str, index: int) -> RequestResult:
    """Download the URL once, counting received bytes and discarding the body."""
    t0 = time.perf_counter()
    try:
        with client.stream("GET", url) as response:
            # client.stream() returns as soon as the response headers are received.
            ttfb_s = time.perf_counter() - t0
            if not response.is_success:
                return RequestResult(
                    index, ok=False, status=response.status_code,
                    error=f"HTTP {response.status_code}",
                )
            size_bytes = 0
            # iter_raw() yields bytes exactly as received, without decompression.
            for chunk in response.iter_raw(CHUNK_SIZE):
                size_bytes += len(chunk)
            duration_s = time.perf_counter() - t0
            return RequestResult(
                index, ok=True, status=response.status_code, size_bytes=size_bytes,
                duration_s=duration_s, ttfb_s=ttfb_s,
            )
    except httpx.TimeoutException:
        # httpx timeouts apply to each network operation, not to the whole request.
        return RequestResult(index, ok=False, error=f"timeout: no data for {client.timeout.read:g} s")
    except httpx.HTTPError as exc:
        return RequestResult(index, ok=False, error=f"{type(exc).__name__}: {exc}")


def run_requests(url: str, count: int, timeout: float) -> Iterator[RequestResult]:
    """Yield results of `count` sequential requests made over one shared client."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        for index in range(1, count + 1):
            yield measure_once(client, url, index)


def summarize(results: Sequence[RequestResult]) -> Summary | None:
    """Compute metrics over successful results; return None if there are none."""
    successful = [r for r in results if r.ok]
    if not successful:
        return None

    durations = [r.duration_s for r in successful]
    total_bytes = sum(r.size_bytes for r in successful)
    # Speed is total bytes / total time, not the mean of per-request speeds: that mean
    # gives every request equal weight regardless of how long it took, so short fast
    # requests count as much as long slow ones. Total bytes / total time is the actual
    # throughput.
    bytes_per_s = total_bytes / sum(durations)

    return Summary(
        succeeded=len(successful),
        total_bytes=total_bytes,
        mean_s=statistics.fmean(durations),
        min_s=min(durations),
        median_s=statistics.median(durations),
        max_s=max(durations),
        mean_ttfb_s=statistics.fmean(r.ttfb_s for r in successful),
        speed_mbit_s=bytes_per_s * 8 / BITS_PER_MBIT,
        speed_mb_s=bytes_per_s / BYTES_PER_MB,
    )


def format_result(result: RequestResult) -> str:
    """Format one result as a table row."""
    if not result.ok:
        return f"{result.index:>3}  {'ERROR':>6}   {result.error}"
    speed_mbit_s = result.size_bytes * 8 / result.duration_s / BITS_PER_MBIT
    return ROW_FORMAT.format(
        result.index,
        result.status,
        f"{result.size_bytes / BYTES_PER_MB:.2f} MB",
        f"{result.duration_s:.3f} s",
        f"{result.ttfb_s:.3f} s",
        f"{speed_mbit_s:.1f} Mbit/s",
    )


def format_summary(summary: Summary | None, attempted: int) -> str:
    """Format the final summary block."""
    if summary is None:
        return f"Successful:        0/{attempted}\nNo successful requests."
    s = summary
    return "\n".join([
        f"Successful:        {s.succeeded}/{attempted}",
        f"Downloaded:        {s.total_bytes / BYTES_PER_MB:.2f} MB ({s.total_bytes:,} bytes)",
        f"Avg request time:  {s.mean_s:.3f} s "
        f"(min {s.min_s:.3f} / median {s.median_s:.3f} / max {s.max_s:.3f})",
        f"Avg TTFB:          {s.mean_ttfb_s:.3f} s",
        f"Download speed:    {s.speed_mbit_s:.1f} Mbit/s ({s.speed_mb_s:.2f} MB/s)",
    ])


def http_url(value: str) -> str:
    """argparse type: accept only http(s) URLs that have a host."""
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise argparse.ArgumentTypeError(f"expected an http:// or https:// URL with a host, got {value!r}")
    return value


def positive_int(value: str) -> int:
    """argparse type: integer >= 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def positive_float(value: str) -> float:
    """argparse type: number > 0."""
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return number


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Measure download speed of a URL.")
    parser.add_argument("url", type=http_url, help="http(s) URL to download, e.g. a large file")
    parser.add_argument(
        "-n", "--requests", type=positive_int, default=10,
        help="number of sequential requests (default: 10)",
    )
    parser.add_argument(
        "--timeout", type=positive_float, default=30.0,
        help="max seconds to wait for each network operation (connect, read, write); "
        "not a limit on the total request time (default: 30)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the measurement and print results. Return the process exit code."""
    args = parse_args(argv)
    print(f"URL:       {args.url}")
    print(f"Requests:  {args.requests}, sequential")
    print()
    print(TABLE_HEADER)

    results: list[RequestResult] = []
    interrupted = False
    try:
        for result in run_requests(args.url, args.requests, args.timeout):
            results.append(result)
            print(format_result(result), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print(f"Interrupted after {len(results)} of {args.requests} requests.")

    summary = summarize(results)
    print(SEPARATOR)
    print(format_summary(summary, len(results)))

    if interrupted:
        return 130
    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
