"""Structured user profile persistence.

Stores a single user profile as JSON in the user data directory so the
assistant can remember and update facts like name, preferences, and goals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.user_data import profile_file


@dataclass
class UserProfile:
    """Typed representation of the user's structured profile."""

    name: str = ""
    email: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "preferences": self.preferences,
            "facts": self.facts,
            "goals": self.goals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        return cls(
            name=data.get("name", ""),
            email=data.get("email", ""),
            preferences=data.get("preferences", {}),
            facts=list(data.get("facts", [])),
            goals=list(data.get("goals", [])),
        )


class UserProfileService:
    """Load, update, and persist a single structured user profile."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or profile_file()
        self._profile: UserProfile | None = None

    def load(self) -> UserProfile:
        """Return the current profile, loading from disk if needed."""
        if self._profile is not None:
            return self._profile
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}
        self._profile = UserProfile.from_dict(data)
        return self._profile

    def save(self, profile: UserProfile) -> None:
        """Persist the profile to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._profile = profile

    def update(self, **kwargs: Any) -> UserProfile:
        """Update specific fields and persist the profile.

        Lists and dicts are merged: new items are appended/added rather than
        replacing existing values, so information accumulates over time.
        """
        profile = self.load()
        for key, value in kwargs.items():
            if key not in {"name", "email", "preferences", "facts", "goals"}:
                continue
            if key in {"preferences"} and isinstance(value, dict):
                getattr(profile, key).update(value)
            elif key in {"facts", "goals"} and isinstance(value, list):
                getattr(profile, key).extend(value)
            else:
                setattr(profile, key, value)
        self.save(profile)
        return profile

    def clear(self) -> None:
        """Remove the persisted profile file."""
        if self.path.is_file():
            self.path.unlink()
        self._profile = None
