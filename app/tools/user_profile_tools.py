from typing import Any

from langchain_core.tools import BaseTool, tool

from app.services.user_profile_service import UserProfileService


def create_user_profile_tools(profile_service: UserProfileService) -> list[BaseTool]:
    """Build tools for reading and updating the structured user profile."""

    @tool
    def get_user_profile() -> str:
        """Return the user's structured profile: name, email, preferences, facts, and goals."""
        profile = profile_service.load()
        parts = [
            f"Name: {profile.name or 'not set'}",
            f"Email: {profile.email or 'not set'}",
            f"Preferences: {profile.preferences or 'none'}",
            f"Facts: {profile.facts or 'none'}",
            f"Goals: {profile.goals or 'none'}",
        ]
        return "\n".join(parts)

    @tool
    def update_user_profile(
        name: str | None = None,
        email: str | None = None,
        preferences: dict[str, Any] | None = None,
        facts: list[str] | None = None,
        goals: list[str] | None = None,
    ) -> str:
        """Update the user's structured profile. Only provided fields are changed.

        Args:
            name: The user's name.
            email: The user's email address.
            preferences: Key/value preferences to add or overwrite.
            facts: New facts about the user to append.
            goals: New goals to append.
        """
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if email is not None:
            kwargs["email"] = email
        if preferences is not None:
            kwargs["preferences"] = preferences
        if facts is not None:
            kwargs["facts"] = facts
        if goals is not None:
            kwargs["goals"] = goals
        profile_service.update(**kwargs)
        return "User profile updated."

    return [get_user_profile, update_user_profile]
