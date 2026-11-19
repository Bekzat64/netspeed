"""End-to-end tests against the local HTTP server from conftest.py."""

from collections.abc import Iterator

import httpx
import pytest

import netspeed
from netspeed import RequestResult, main, measure_once, run_requests, summarize

TIMEOUT_S = 5.0


def test_makes_exactly_n_requests_and_counts_all_bytes(
    base_url: str, request_paths: list[str], payload_size: int
) -> None:
    count = 3

    results = list(run_requests(f"{base_url}/payload", count, TIMEOUT_S))
    summary = summarize(results)

    assert request_paths == ["/payload"] * count
    assert all(r.ok and r.status == 200 for r in results)
    assert summary is not None
    assert summary.total_bytes == count * payload_size


def test_404_fails_every_request_and_main_returns_1(
    base_url: str, request_paths: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([f"{base_url}/missing", "-n", "2"])

    assert exit_code == 1
    assert len(request_paths) == 2
    output = capsys.readouterr().out
    assert output.count("ERROR   HTTP 404") == 2
    assert "Successful:        0/2" in output


def test_redirect_is_followed(base_url: str, request_paths: list[str], payload_size: int) -> None:
    [result] = run_requests(f"{base_url}/redirect", 1, TIMEOUT_S)

    assert result.ok
    assert result.status == 200
    assert result.size_bytes == payload_size
    assert request_paths == ["/redirect", "/payload"]


def test_truncated_body_fails_and_is_not_counted(base_url: str, payload_size: int) -> None:
    with httpx.Client(timeout=TIMEOUT_S) as client:
        full = measure_once(client, f"{base_url}/payload", 1)
        truncated = measure_once(client, f"{base_url}/truncated", 2)

    summary = summarize([full, truncated])

    assert full.ok
    assert not truncated.ok
    assert truncated.error  # some message from httpx; exact text is not part of the contract
    assert summary is not None
    assert summary.succeeded == 1
    assert summary.total_bytes == payload_size


def test_timeout_fails_the_request_and_main_returns_1(
    base_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # /slow waits 0.5 s before sending headers, longer than the 0.1 s timeout.
    exit_code = main([f"{base_url}/slow", "-n", "1", "--timeout", "0.1"])

    assert exit_code == 1
    assert "ERROR   timeout: no data for 0.1 s" in capsys.readouterr().out


def test_ctrl_c_prints_summary_and_returns_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_requests(url: str, count: int, timeout: float) -> Iterator[RequestResult]:
        yield RequestResult(
            1, ok=True, status=200, size_bytes=1_000_000, duration_s=0.5, ttfb_s=0.1
        )
        raise KeyboardInterrupt

    # main() looks up run_requests in the netspeed module at call time, so this replaces it.
    monkeypatch.setattr(netspeed, "run_requests", fake_run_requests)

    exit_code = main(["http://example.com/file"])

    assert exit_code == 130
    output = capsys.readouterr().out
    assert "Interrupted after 1 of 10 requests." in output
    assert "Successful:        1/1" in output


@pytest.mark.parametrize("url", ["ftp://example.com/file", "http://:80", "example.com"])
def test_invalid_url_exits_with_code_2(url: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([url])

    assert exc_info.value.code == 2
