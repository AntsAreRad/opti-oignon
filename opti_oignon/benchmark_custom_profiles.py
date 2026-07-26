#!/usr/bin/env python3
"""
Benchmark Custom Profiles

CRUD operations for user-defined benchmark profiles. Custom profiles
are stored in ``data/custom_benchmark_profiles.yaml`` and merged with
built-in profiles at runtime so they appear alongside defaults in the
Run tab and profile selectors.

Each custom profile specifies:
  - Unique ID (auto-generated or user-supplied)
  - Human-readable name and description
  - List of question categories (subset of available categories)
  - Weight preset name OR custom weight values
  - Optional runner overrides (timeout, max_response_tokens, etc.)
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_CUSTOM_PROFILES_PATH = _DATA_DIR / "custom_benchmark_profiles.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, returning empty dict")
        return {}
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_yaml(data: dict, path: Path) -> bool:
    """Save a dict to a YAML file."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, cannot save")
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
        return True
    except OSError as exc:
        logger.error("Failed to save custom profiles: %s", exc)
        return False


def _validate_profile_fields(fields: dict) -> None:
    """Validate custom-profile numeric fields.

    Only validates keys that are present and non-None, so it serves both
    create (full set) and partial update. Raises ValueError on the first
    violation, which the routes map to HTTP 400.
    """
    if fields.get("timeout") is not None:
        t = fields["timeout"]
        if not isinstance(t, int) or isinstance(t, bool) or not (1 <= t <= 600):
            raise ValueError("timeout must be an integer between 1 and 600 seconds")
    if fields.get("max_response_tokens") is not None:
        mt = fields["max_response_tokens"]
        if not isinstance(mt, int) or isinstance(mt, bool) or not (1 <= mt <= 32768):
            raise ValueError(
                "max_response_tokens must be an integer between 1 and 32768"
            )
    if fields.get("expected_length_range") is not None:
        r = fields["expected_length_range"]
        if (not isinstance(r, (list, tuple)) or len(r) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in r)):
            raise ValueError(
                "expected_length_range must be a [min, max] integer pair"
            )
        if not (0 <= r[0] < r[1]):
            raise ValueError("expected_length_range must satisfy 0 <= min < max")
    if fields.get("custom_weights") is not None:
        cw = fields["custom_weights"]
        required = {"accuracy", "code", "structure", "speed"}
        if not isinstance(cw, dict) or required - set(cw.keys()):
            raise ValueError(
                "custom_weights must include accuracy, code, structure, speed"
            )
        total = 0.0
        for k in required:
            v = cw[k]
            if (not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not (0.0 <= float(v) <= 1.0)):
                raise ValueError(f"custom_weights['{k}'] must be a number in [0, 1]")
            total += float(v)
        if total <= 0.0:
            raise ValueError("custom_weights must sum to more than zero")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CustomProfile:
    """A user-defined benchmark profile."""
    profile_id: str = ""
    name: str = ""
    description: str = ""
    categories: list[str] = field(default_factory=list)
    weight_preset: str = "balanced"
    custom_weights: dict[str, float] | None = None
    timeout: int = 45
    max_response_tokens: int = 800
    expected_length_range: list[int] = field(default_factory=lambda: [10, 600])
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "categories": self.categories,
            "weight_preset": self.weight_preset,
            "timeout": self.timeout,
            "max_response_tokens": self.max_response_tokens,
            "expected_length_range": self.expected_length_range,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.custom_weights:
            result["custom_weights"] = self.custom_weights
        return result

    def to_profile_entry(self) -> dict:
        """Convert to the format used by benchmark_evaluator profiles."""
        entry: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "categories": self.categories,
            "weight_preset": self.weight_preset,
            "timeout": self.timeout,
            "max_response_tokens": self.max_response_tokens,
            "expected_length_range": self.expected_length_range,
            "custom": True,
        }
        if self.custom_weights:
            entry["custom_weights"] = self.custom_weights
        return entry

    @staticmethod
    def from_dict(data: dict) -> "CustomProfile":
        return CustomProfile(
            profile_id=data.get("profile_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            categories=data.get("categories", []),
            weight_preset=data.get("weight_preset", "balanced"),
            custom_weights=data.get("custom_weights"),
            timeout=data.get("timeout", 45),
            max_response_tokens=data.get("max_response_tokens", 800),
            expected_length_range=data.get(
                "expected_length_range", [10, 600]
            ),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )


# ---------------------------------------------------------------------------
# Custom Profile Store
# ---------------------------------------------------------------------------

class CustomProfileStore:
    """CRUD store for user-defined benchmark profiles.

    Thread-safe. Profiles are persisted to YAML on every mutation.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _CUSTOM_PROFILES_PATH
        self._profiles: dict[str, CustomProfile] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load profiles from disk."""
        data = _load_yaml(self._path)
        profiles_raw = data.get("profiles", {})
        self._profiles = {}
        for pid, pdata in profiles_raw.items():
            # Custom profile ids are always custom_*; reject a
            # hand-edited entry with a builtin id, which would otherwise
            # shadow the builtin at the evaluator merge (the PIP-03 class).
            if not str(pid).startswith("custom_"):
                logger.warning(
                    "Ignoring custom profile with non-custom id '%s'", pid,
                )
                continue
            pdata["profile_id"] = pid
            self._profiles[pid] = CustomProfile.from_dict(pdata)

    def _persist(self) -> bool:
        """Save all profiles to disk."""
        data = {
            "profiles": {
                pid: p.to_dict() for pid, p in self._profiles.items()
            }
        }
        return _save_yaml(data, self._path)

    # -- CRUD --

    def list_profiles(self) -> list[CustomProfile]:
        """List all custom profiles."""
        with self._lock:
            return list(self._profiles.values())

    def get(self, profile_id: str) -> CustomProfile | None:
        """Get a single profile by ID."""
        with self._lock:
            return self._profiles.get(profile_id)

    def create(
        self,
        name: str,
        description: str = "",
        categories: list[str] | None = None,
        weight_preset: str = "balanced",
        custom_weights: dict[str, float] | None = None,
        timeout: int = 45,
        max_response_tokens: int = 800,
        expected_length_range: list[int] | None = None,
    ) -> CustomProfile:
        """Create a new custom profile.

        Args:
            name: Human-readable profile name (max 64 chars, must be unique).
            description: Description of the profile.
            categories: List of question categories to include.
            weight_preset: Named weight preset or 'custom'.
            custom_weights: Custom weight values when weight_preset='custom'.
            timeout: Per-question timeout in seconds.
            max_response_tokens: Max tokens for model response.
            expected_length_range: [min, max] expected response length.

        Returns:
            The newly created CustomProfile.

        Raises:
            ValueError: If name is empty, too long, or already exists.
        """
        stripped = name.strip()
        if not stripped:
            raise ValueError("Profile name cannot be empty")
        if len(stripped) > 64:
            raise ValueError(
                f"Profile name too long ({len(stripped)} chars, max 64)"
            )

        # Bound the numeric fields before persisting.
        _validate_profile_fields({
            "timeout": timeout,
            "max_response_tokens": max_response_tokens,
            "expected_length_range": expected_length_range or [10, 600],
            "custom_weights": custom_weights,
        })

        with self._lock:
            for existing in self._profiles.values():
                if existing.name.lower() == stripped.lower():
                    raise ValueError(
                        f"Profile name '{stripped}' already exists"
                    )

            profile_id = f"custom_{uuid.uuid4().hex[:8]}"
            now = time.time()

            profile = CustomProfile(
                profile_id=profile_id,
                name=stripped,
                description=description,
                categories=categories or [],
                weight_preset=weight_preset,
                custom_weights=custom_weights,
                timeout=timeout,
                max_response_tokens=max_response_tokens,
                expected_length_range=expected_length_range or [10, 600],
                created_at=now,
                updated_at=now,
            )

            self._profiles[profile_id] = profile
            self._persist()

        logger.info("Created custom profile: %s (%s)", profile_id, stripped)
        return profile

    def update(
        self,
        profile_id: str,
        updates: dict,
    ) -> CustomProfile | None:
        """Update an existing custom profile.

        Args:
            profile_id: ID of the profile to update.
            updates: Dict of fields to update.

        Returns:
            Updated profile, or None if not found.

        Raises:
            ValueError: If name update violates constraints.
        """
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                return None

            # Validate name if being updated
            if "name" in updates:
                new_name = str(updates["name"]).strip()
                if not new_name:
                    raise ValueError("Profile name cannot be empty")
                if len(new_name) > 64:
                    raise ValueError(
                        f"Profile name too long ({len(new_name)} chars, max 64)"
                    )
                for pid, existing in self._profiles.items():
                    if pid != profile_id and existing.name.lower() == new_name.lower():
                        raise ValueError(
                            f"Profile name '{new_name}' already exists"
                        )
                updates["name"] = new_name

            # Validate any numeric fields present in the update
            # (custom_weights here must also carry the four keys, aligning
            # update with the create-path check).
            _validate_profile_fields(updates)

            allowed = {
                "name", "description", "categories", "weight_preset",
                "custom_weights", "timeout", "max_response_tokens",
                "expected_length_range",
            }

            for key, value in updates.items():
                if key in allowed:
                    setattr(profile, key, value)

            profile.updated_at = time.time()
            self._persist()

        logger.info("Updated custom profile: %s", profile_id)
        return profile

    def delete(self, profile_id: str) -> bool:
        """Delete a custom profile.

        Args:
            profile_id: ID of the profile to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if profile_id not in self._profiles:
                return False
            del self._profiles[profile_id]
            self._persist()

        logger.info("Deleted custom profile: %s", profile_id)
        return True

    def count(self) -> int:
        """Return the number of custom profiles."""
        with self._lock:
            return len(self._profiles)

    def reload(self) -> None:
        """Reload profiles from disk."""
        with self._lock:
            self._load()

    # -- Integration helpers --

    def as_profiles_dict(self) -> dict[str, dict]:
        """Return all custom profiles in the evaluator profile format.

        Returns:
            Dict mapping profile_id to profile entry dict (same shape as
            entries in benchmark_profiles.yaml).
        """
        with self._lock:
            return {
                pid: p.to_profile_entry()
                for pid, p in self._profiles.items()
            }

    def get_question_preview(
        self,
        categories: list[str],
        available_questions: dict[str, list] | None = None,
    ) -> dict:
        """Preview which questions would be included for given categories.

        Args:
            categories: List of category names.
            available_questions: Pre-loaded questions dict (optional).

        Returns:
            Dict with category_counts and total.
        """
        if available_questions is None:
            try:
                from opti_oignon.benchmark_evaluator import load_questions
                available_questions = load_questions()
            except ImportError:
                return {"category_counts": {}, "total": 0}

        counts: dict[str, int] = {}
        total = 0
        for cat in categories:
            qs = available_questions.get(cat, [])
            counts[cat] = len(qs)
            total += len(qs)

        return {"category_counts": counts, "total": total}



# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

try:
    custom_profile_store = CustomProfileStore()
    CUSTOM_PROFILES_AVAILABLE = True
except Exception as e:
    logger.warning("CustomProfileStore init failed: %s", e)
    custom_profile_store = None  # type: ignore[assignment]
    CUSTOM_PROFILES_AVAILABLE = False
