"""Tests for RunRecord serialization and reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from recension import (
    Budget,
    CandidateRecord,
    Provenance,
    RoundRecord,
    RunRecord,
    SliceScore,
    TextArtifact,
)


def make_record() -> RunRecord:
    artifact = TextArtifact.from_text("v1 text\n", name="demo")
    root_id = artifact.current().version_id
    v2 = artifact.commit("v2 text\n", Provenance(diagnosis="too vague"))
    return RunRecord(
        artifact=artifact.to_dict(),
        objective_name="exact_match",
        model_graded=False,
        seed=7,
        budget=Budget(max_model_calls=50).to_dict(),
        baseline_version_id=root_id,
        baseline_score=0.25,
        rounds=[
            RoundRecord(
                round_index=1,
                incumbent_version_id=root_id,
                incumbent_validation_score=0.25,
                train_score=0.5,
                failure_example_ids=("t1", "t2"),
                diagnosis="too vague",
                candidates=(
                    CandidateRecord(
                        candidate_id="r1-c1",
                        text="v2 text\n",
                        validation_score=0.75,
                        diff="--- a\n+++ b\n",
                        accepted=True,
                    ),
                    CandidateRecord(
                        candidate_id="r1-c2",
                        text="loser\n",
                        validation_score=0.1,
                        diff="",
                        leakage_flags=("verbatim_validation_span: ...",),
                    ),
                ),
                accepted_version_id=v2.version_id,
                model_calls_used=9,
                elapsed_seconds=0.01,
            )
        ],
        final_version_id=v2.version_id,
        final_score=0.75,
        total_model_calls=11,
        stopped_reason="completed",
        started_at="2026-06-10T00:00:00+00:00",
        finished_at="2026-06-10T00:00:01+00:00",
    )


def test_json_roundtrip() -> None:
    record = make_record()
    restored = RunRecord.from_json(record.to_json())
    assert restored == record


def test_save_and_load(tmp_path: Path) -> None:
    record = make_record()
    path = tmp_path / "record.json"
    record.save(path)
    assert RunRecord.load(path) == record


def test_restored_artifact_supports_diff() -> None:
    record = make_record()
    artifact = record.restored_artifact()
    out = artifact.diff(record.baseline_version_id, record.final_version_id)
    assert "-v1 text" in out
    assert "+v2 text" in out


def test_verify_passes_on_intact_record() -> None:
    record = make_record()
    assert record.verify() == []


def test_verify_catches_tampered_embedded_artifact() -> None:
    record = make_record()
    # Forge a version's text in the embedded artifact without fixing its id.
    record.artifact["versions"][1]["text"] = "forged text\n"
    problems = record.verify()
    assert problems
    assert "content hash" in problems[0]


def test_fingerprint_is_stable_and_change_sensitive() -> None:
    record = make_record()
    fp = record.fingerprint()
    assert record.fingerprint() == fp  # stable for the same record
    assert RunRecord.from_json(record.to_json()).fingerprint() == fp  # survives a round trip
    record.final_score = 0.99  # any field change moves the fingerprint
    assert record.fingerprint() != fp


def test_partial_record_with_nan_score_serializes_to_valid_json() -> None:
    # A budget-exhausted partial record can carry an uncomputed NaN score; the
    # serialized form must still be valid, interoperable JSON.
    record = make_record()
    record.baseline_score = float("nan")
    payload = record.to_json()
    assert "NaN" not in payload
    assert json.loads(payload)["baseline_score"] is None  # strict parse succeeds
    assert len(record.fingerprint()) == 64  # canonical JSON is computable


def test_hmac_signature_roundtrip() -> None:
    record = make_record()
    sig = record.sign("s3cret")
    assert record.verify_signature("s3cret", sig)
    assert not record.verify_signature("wrong-key", sig)
    assert not record.verify_signature("s3cret", "deadbeef")


def test_slice_score_regressed_property_and_summary() -> None:
    record = make_record()
    record.slice_scores = [
        SliceScore(slice="us", n=10, baseline_score=0.6, final_score=0.8),
        SliceScore(slice="eu", n=8, baseline_score=0.7, final_score=0.5),
    ]
    assert not record.slice_scores[0].regressed
    assert record.slice_scores[1].regressed
    text = record.summary()
    assert "slices:" in text
    assert "eu (n=8): 0.7000 -> 0.5000  [REGRESSED]" in text
    assert RunRecord.from_json(record.to_json()) == record


def test_summary_degrades_on_missing_accepted_candidate() -> None:
    # A complete record always has the accepted candidate, but summary() must
    # not crash on a hand-edited/truncated one whose accepted_version_id is set
    # while no candidate is flagged accepted.
    record = make_record()
    broken_round = replace(
        record.rounds[0],
        candidates=tuple(replace(c, accepted=False) for c in record.rounds[0].candidates),
    )
    record.rounds[0] = broken_round
    text = record.summary()
    assert "accepted candidate not found" in text


def test_summary_is_reviewable() -> None:
    text = make_record().summary()
    assert "baseline" in text
    assert "0.2500" in text
    assert "diagnosis: too vague" in text
    assert "ACCEPTED" in text
    assert "rejected" in text
    assert "verbatim_validation_span" in text
    assert "0.2500 -> 0.7500" in text
    assert "stopped: completed" in text


def test_record_is_complete_for_an_external_reviewer() -> None:
    """A reviewer must be able to reconstruct every decision from the record alone."""
    data = make_record().to_dict()
    # The artifact history, every candidate (with text and score), the
    # diagnosis, the flags, and the budget are all present in the raw dict.
    assert data["artifact"]["versions"][1]["provenance"]["diagnosis"] == "too vague"
    round_data = data["rounds"][0]
    assert {c["candidate_id"] for c in round_data["candidates"]} == {"r1-c1", "r1-c2"}
    assert all("text" in c and "validation_score" in c for c in round_data["candidates"])
    assert data["budget"]["max_model_calls"] == 50
    assert data["model_graded"] is False
