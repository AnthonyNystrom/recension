"""Versioned text artifacts with provenance.

A :class:`TextArtifact` holds the text being optimized (a prompt, a context
template, a skill file) together with an append-only, linear version history.
Every version after the root carries a :class:`Provenance`: the diagnosis that
motivated the change, the scores that justified it, the sibling candidates that
were rejected, and a unified diff against the parent. A reviewer can
reconstruct every accepted edit from the artifact alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path
from typing import Any

from .exceptions import ArtifactError

__all__ = ["Provenance", "RejectedCandidate", "TextArtifact", "Version"]


@dataclass(frozen=True)
class RejectedCandidate:
    """A sibling candidate that lost to the accepted version.

    Kept in full (text included) so the comparison that justified the accepted
    edit can be reproduced later.
    """

    candidate_id: str
    text: str
    score: float | None
    leakage_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Provenance:
    """Why a version exists.

    Attributes:
        diagnosis: Free-text hypothesis about what in the parent text caused
            the observed failures (or a note such as a rollback reason).
        failure_example_ids: Ids of the train examples whose failures
            motivated the change.
        incumbent_score: Held-out score of the parent version, if measured.
        candidate_score: Held-out score of this version, if measured.
        rejected_candidates: Sibling candidates considered in the same round,
            with their scores and any leakage flags.
        diff: Unified diff against the parent text. Always computed by
            :meth:`TextArtifact.commit`, never supplied by the caller, so it
            cannot drift from the actual texts.
    """

    diagnosis: str
    failure_example_ids: tuple[str, ...] = ()
    incumbent_score: float | None = None
    candidate_score: float | None = None
    rejected_candidates: tuple[RejectedCandidate, ...] = ()
    diff: str = ""


@dataclass(frozen=True)
class Version:
    """One immutable entry in an artifact's history."""

    version_id: str
    parent_id: str | None
    text: str
    created_at: str
    provenance: Provenance | None = None


def _version_id(parent_id: str | None, text: str) -> str:
    # DESIGN NOTE: version ids are content-addressed (parent id + text) rather
    # than random, so a seeded run against MockModel produces byte-identical
    # version ids, part of the determinism guarantee.
    digest = hashlib.sha256(f"{parent_id}\x00{text}".encode()).hexdigest()
    return digest[:12]


class TextArtifact:
    """A text under optimization, with its full version history.

    The history is linear and append-only: each version has exactly one
    parent, and nothing is ever rewritten or deleted. ``rollback`` therefore
    *appends* a new version whose text restores an earlier one, rather than
    moving a pointer backwards. The record of having tried and reverted is
    itself part of the audit trail.
    """

    def __init__(self, versions: list[Version], name: str = "artifact") -> None:
        """Build an artifact from an existing linear history.

        Most callers should use :meth:`from_text` or :meth:`from_file`.

        Raises:
            ArtifactError: If ``versions`` is empty or not a linear chain.
        """
        if not versions:
            raise ArtifactError("an artifact needs at least one version")
        for i, version in enumerate(versions):
            expected_parent = None if i == 0 else versions[i - 1].version_id
            if version.parent_id != expected_parent:
                raise ArtifactError(
                    f"versions do not form a linear chain at {version.version_id!r}"
                )
        self.name = name
        self._versions: list[Version] = list(versions)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, name: str = "artifact") -> TextArtifact:
        """Create a new artifact whose root version holds ``text``."""
        root = Version(
            version_id=_version_id(None, text),
            parent_id=None,
            text=text,
            created_at=datetime.now(UTC).isoformat(),
        )
        return cls([root], name=name)

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> TextArtifact:
        """Create a new artifact from the contents of a text file.

        The artifact name defaults to the file's stem.
        """
        p = Path(path)
        return cls.from_text(p.read_text(encoding="utf-8"), name=name or p.stem)

    # -- reading ------------------------------------------------------------

    def current(self) -> Version:
        """The latest version (the incumbent)."""
        return self._versions[-1]

    @property
    def text(self) -> str:
        """The current version's text."""
        return self.current().text

    def history(self) -> list[Version]:
        """All versions, root first, current last."""
        return list(self._versions)

    def get(self, version_id: str) -> Version:
        """Look up a version by id.

        Raises:
            ArtifactError: If no version has that id.
        """
        for version in self._versions:
            if version.version_id == version_id:
                return version
        raise ArtifactError(f"unknown version id {version_id!r} in artifact {self.name!r}")

    def diff(self, version_a: str, version_b: str) -> str:
        """Unified diff between two versions' texts, by version id."""
        a, b = self.get(version_a), self.get(version_b)
        return _diff_texts(a.text, b.text, version_a, version_b)

    def verify(self) -> list[str]:
        """Check content-addressing integrity of the version history.

        Version ids are a hash of ``(parent_id, text)``, so editing a version's
        text or id after the fact, or breaking the parent chain, is detectable.
        Returns a list of human-readable problems, empty when the history is
        intact. This is the self-contained tamper-evidence behind
        :meth:`recension.record.RunRecord.verify`.
        """
        problems: list[str] = []
        for i, version in enumerate(self._versions):
            expected_parent = None if i == 0 else self._versions[i - 1].version_id
            if version.parent_id != expected_parent:
                problems.append(
                    f"version {version.version_id!r} has parent {version.parent_id!r}, "
                    f"expected {expected_parent!r}"
                )
            expected_id = _version_id(version.parent_id, version.text)
            if version.version_id != expected_id:
                problems.append(
                    f"version {version.version_id!r} does not match its content hash "
                    f"({expected_id!r}); the text or id may have been altered"
                )
        return problems

    # -- writing ------------------------------------------------------------

    def commit(self, text: str, provenance: Provenance) -> Version:
        """Append a new version with ``text`` and ``provenance``.

        The diff against the parent is computed here and written into the
        stored provenance; any caller-supplied ``provenance.diff`` is ignored.

        Raises:
            ArtifactError: If ``text`` is identical to the current text
                (a no-op commit would corrupt the history's meaning).
        """
        parent = self.current()
        if text == parent.text:
            raise ArtifactError("refusing no-op commit: text is identical to the current version")
        new_id = _version_id(parent.version_id, text)
        stored = replace(provenance, diff=_diff_texts(parent.text, text, parent.version_id, new_id))
        version = Version(
            version_id=new_id,
            parent_id=parent.version_id,
            text=text,
            created_at=datetime.now(UTC).isoformat(),
            provenance=stored,
        )
        self._versions.append(version)
        return version

    def rollback(self, version_id: str) -> Version:
        """Restore an earlier version's text by appending a new version.

        Raises:
            ArtifactError: If ``version_id`` is unknown or already current.
        """
        target = self.get(version_id)
        if target.version_id == self.current().version_id:
            raise ArtifactError(f"version {version_id!r} is already current")
        return self.commit(
            target.text,
            Provenance(diagnosis=f"rollback to version {version_id}"),
        )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, suitable for embedding in a run record.

        JSON-pure (lists, not tuples), so a serialize/deserialize round trip
        is the identity.
        """
        return {
            "name": self.name,
            "versions": [_version_to_dict(v) for v in self._versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextArtifact:
        """Inverse of :meth:`to_dict`."""
        versions = [
            Version(
                version_id=v["version_id"],
                parent_id=v["parent_id"],
                text=v["text"],
                created_at=v["created_at"],
                provenance=_provenance_from_dict(v["provenance"]) if v["provenance"] else None,
            )
            for v in data["versions"]
        ]
        return cls(versions, name=data["name"])

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the full artifact (history included) to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> TextArtifact:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(payload))


def _version_to_dict(version: Version) -> dict[str, Any]:
    provenance = None
    if version.provenance is not None:
        p = version.provenance
        provenance = {
            "diagnosis": p.diagnosis,
            "failure_example_ids": list(p.failure_example_ids),
            "incumbent_score": p.incumbent_score,
            "candidate_score": p.candidate_score,
            "rejected_candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "text": c.text,
                    "score": c.score,
                    "leakage_flags": list(c.leakage_flags),
                }
                for c in p.rejected_candidates
            ],
            "diff": p.diff,
        }
    return {
        "version_id": version.version_id,
        "parent_id": version.parent_id,
        "text": version.text,
        "created_at": version.created_at,
        "provenance": provenance,
    }


def _provenance_from_dict(data: dict[str, Any]) -> Provenance:
    return Provenance(
        diagnosis=data["diagnosis"],
        failure_example_ids=tuple(data["failure_example_ids"]),
        incumbent_score=data["incumbent_score"],
        candidate_score=data["candidate_score"],
        rejected_candidates=tuple(
            RejectedCandidate(
                candidate_id=c["candidate_id"],
                text=c["text"],
                score=c["score"],
                leakage_flags=tuple(c["leakage_flags"]),
            )
            for c in data["rejected_candidates"]
        ),
        diff=data["diff"],
    )


def _diff_texts(a: str, b: str, label_a: str, label_b: str) -> str:
    # DESIGN NOTE: stdlib `unified_diff` does not emit git's "\ No newline at
    # end of file" marker, so a content line from a file with no trailing
    # newline runs straight into the next line when joined. We add the marker
    # ourselves, matching `git diff`, so artifact diffs render cleanly in the
    # record, the CLI, the docs, and the demo.
    out: list[str] = []
    for line in unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile=label_a,
        tofile=label_b,
    ):
        out.append(line)
        # Only a content line (' ', '-', '+') at a no-newline file end lacks a
        # trailing newline; the `---`/`+++`/`@@` headers always end in one.
        if line.startswith((" ", "-", "+")) and not line.endswith("\n"):
            out.append("\n\\ No newline at end of file\n")
    return "".join(out)
