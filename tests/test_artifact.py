"""Tests for TextArtifact versioning, diffs, rollback, and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from recension import ArtifactError, Provenance, RejectedCandidate, TextArtifact


def make_artifact() -> TextArtifact:
    return TextArtifact.from_text("Classify the sentiment.\n", name="clf")


def test_root_version_has_no_parent_or_provenance() -> None:
    artifact = make_artifact()
    root = artifact.current()
    assert root.parent_id is None
    assert root.provenance is None
    assert artifact.text == "Classify the sentiment.\n"
    assert artifact.history() == [root]


def test_commit_appends_linear_history_with_computed_diff() -> None:
    artifact = make_artifact()
    root = artifact.current()
    v2 = artifact.commit(
        "Classify the sentiment as positive or negative.\n",
        Provenance(
            diagnosis="label space was unstated",
            failure_example_ids=("ex-1",),
            incumbent_score=0.5,
            candidate_score=0.8,
            rejected_candidates=(
                RejectedCandidate(candidate_id="cand-2", text="alt text", score=0.6),
            ),
        ),
    )
    assert v2.parent_id == root.version_id
    assert artifact.current() == v2
    assert [v.version_id for v in artifact.history()] == [root.version_id, v2.version_id]
    assert v2.provenance is not None
    assert "-Classify the sentiment." in v2.provenance.diff
    assert "+Classify the sentiment as positive or negative." in v2.provenance.diff


def test_commit_ignores_caller_supplied_diff() -> None:
    artifact = make_artifact()
    v2 = artifact.commit("New text.\n", Provenance(diagnosis="x", diff="bogus"))
    assert v2.provenance is not None
    assert v2.provenance.diff != "bogus"
    assert "+New text." in v2.provenance.diff


def test_noop_commit_raises() -> None:
    artifact = make_artifact()
    with pytest.raises(ArtifactError, match="no-op"):
        artifact.commit(artifact.text, Provenance(diagnosis="nothing changed"))


def test_version_ids_are_deterministic() -> None:
    a, b = make_artifact(), make_artifact()
    assert a.current().version_id == b.current().version_id
    va = a.commit("Second.\n", Provenance(diagnosis="d"))
    vb = b.commit("Second.\n", Provenance(diagnosis="d"))
    assert va.version_id == vb.version_id


def test_diff_between_arbitrary_versions() -> None:
    artifact = make_artifact()
    root = artifact.current()
    v2 = artifact.commit("Second.\n", Provenance(diagnosis="d"))
    out = artifact.diff(root.version_id, v2.version_id)
    assert "-Classify the sentiment." in out
    assert "+Second." in out


def test_diff_marks_missing_final_newline_so_lines_do_not_run_together() -> None:
    # A source text with no trailing newline must not let the removed line run
    # into the next added line; we emit git's "\ No newline at end of file".
    artifact = TextArtifact.from_text("First line, no newline.")
    v2 = artifact.commit("First line, no newline.\nSecond line.", Provenance(diagnosis="d"))
    assert v2.provenance is not None
    diff = v2.provenance.diff
    assert "\\ No newline at end of file" in diff
    # The removed line ends cleanly; it does not concatenate with the added line.
    assert "-First line, no newline.\n" in diff
    assert "-First line, no newline.+" not in diff


def test_diff_with_trailing_newlines_has_no_marker() -> None:
    artifact = TextArtifact.from_text("a\n")
    v2 = artifact.commit("b\n", Provenance(diagnosis="d"))
    assert v2.provenance is not None
    assert "No newline at end of file" not in v2.provenance.diff


def test_rollback_appends_rather_than_rewriting() -> None:
    artifact = make_artifact()
    root = artifact.current()
    artifact.commit("Second.\n", Provenance(diagnosis="d"))
    restored = artifact.rollback(root.version_id)
    assert artifact.text == root.text
    assert len(artifact.history()) == 3
    assert restored.provenance is not None
    assert root.version_id in restored.provenance.diagnosis


def test_rollback_to_current_raises() -> None:
    artifact = make_artifact()
    with pytest.raises(ArtifactError, match="already current"):
        artifact.rollback(artifact.current().version_id)


def test_unknown_version_raises() -> None:
    artifact = make_artifact()
    with pytest.raises(ArtifactError, match="unknown version"):
        artifact.get("nope")


def test_json_roundtrip_preserves_everything() -> None:
    artifact = make_artifact()
    artifact.commit(
        "Second.\n",
        Provenance(
            diagnosis="d",
            failure_example_ids=("ex-1", "ex-2"),
            incumbent_score=0.25,
            candidate_score=0.75,
            rejected_candidates=(
                RejectedCandidate("cand-2", "loser", 0.1, leakage_flags=("verbatim_span",)),
            ),
        ),
    )
    restored = TextArtifact.from_json(artifact.to_json())
    assert restored.name == artifact.name
    assert restored.history() == artifact.history()


def test_from_file(tmp_path: Path) -> None:
    p = tmp_path / "prompt.txt"
    p.write_text("From a file.\n", encoding="utf-8")
    artifact = TextArtifact.from_file(p)
    assert artifact.name == "prompt"
    assert artifact.text == "From a file.\n"
