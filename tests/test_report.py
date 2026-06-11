"""Tests for the standalone HTML audit report."""

from __future__ import annotations

from test_record import make_record

from recension import render_report


def test_report_is_standalone_html_with_key_fields() -> None:
    html = render_report(make_record())
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "<style>" in html  # inline CSS, no external assets
    assert "recension audit: demo" in html
    assert "Round 1" in html
    assert "too vague" in html  # the diagnosis
    assert "accepted" in html
    assert "integrity: verified" in html or "verified" in html


def test_report_renders_the_accepted_diff() -> None:
    html = render_report(make_record())
    assert 'class="diff"' in html


def test_report_escapes_html_in_record_content() -> None:
    record = make_record()
    record.artifact["name"] = "<script>alert(1)</script>"
    html = render_report(record)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_flags_failed_integrity() -> None:
    record = make_record()
    record.artifact["versions"][1]["text"] = "tampered\n"  # break the version chain
    html = render_report(record)
    assert "FAILED" in html


def test_report_does_not_color_diff_file_headers() -> None:
    # The unified-diff --- / +++ file headers are not added/removed content and
    # must not be rendered green/red.
    html = render_report(make_record())  # its accepted diff is "--- a\n+++ b\n"
    assert '<span class="del">---' not in html
    assert '<span class="add">+++' not in html
