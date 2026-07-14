#!/usr/bin/env python3
"""
Backup Manager -- Opti-Oignon S121

Full configuration backup and restore system. Exports and imports
platform configuration as a single JSON file (.oo-backup.json).

Supported sections:
  - presets: user task presets
  - system_presets: system-level preset assignments
  - routing: smart router configuration
  - learned_routing: learned router weights and config
  - plugins: plugin states and per-plugin configs
  - rag_metadata: RAG collection names and document metadata (NOT vectors)
  - compression: conversation compressor settings
  - telemetry: telemetry pipeline + history settings
  - sandbox: sandbox isolation configuration
  - theme: user preferences and theme settings
  - model_profiles: model profile assignments
  - cascading: cascading inference tier config
  - speculative: speculative decoding pairings and config
  - benchmarks: custom benchmark profiles
  - semantic_cache: semantic cache settings (config only, never content)
  - benchmark_auto_trigger: benchmark auto-trigger settings
  - humanizer: humanizer settings (config only, never feedback data)
  - fine_tune: fine-tune config and the variants registry (never A/B results)
  - custom_pipelines: user-authored agent pipelines (builtins excluded)
  - execution_pipelines: user-authored execution pipelines (builtins excluded)
  - projects_settings: projects feature settings (never project content)

Import strategies:
  - merge: keep existing values, add missing ones
  - replace: overwrite with backup values

Schema version 1.0 for forward compatibility.

Forward compatibility (S220 BK-06): sections grow additively and the
schema version stays 1.0. An old backup imports cleanly on a newer
install (all of its sections are still known). A backup written by a
newer install can carry sections an older install does not know;
validate_backup on the older install rejects it explicitly, naming each
unknown section. The asymmetry is accepted and documented: the failure
is honest and fail-secure, never a silent partial import.
"""

import base64
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PQC signature support (S129)
# ---------------------------------------------------------------------------

try:
    from opti_oignon.pqc_signatures import (
        PQC_AVAILABLE as _PQC_LIB_AVAILABLE,
    )
    from opti_oignon.pqc_signatures import (
        assert_pqc_posture,
        is_pqc_enabled,
        load_pqc_keypair,
        pqc_keypair_exists,
        pqc_required,
    )
    from opti_oignon.pqc_signatures import (
        sign_backup as _pqc_sign,
    )
    from opti_oignon.pqc_signatures import (
        verify_backup as _pqc_verify,
    )
except ImportError:
    _PQC_LIB_AVAILABLE = False

    def is_pqc_enabled() -> bool:
        return False

    def pqc_keypair_exists(path=None) -> bool:
        return False

    def pqc_required() -> bool:
        # Unreachable: the refusal below fires first. Present so that a caller
        # asking the requirement finds a name rather than a NameError, and so
        # that the shape of this fallback matches the module it stands in for.
        return False

    def assert_pqc_posture() -> None:
        # The signing module itself would not import. That is a broken tree, not
        # a posture -- and it must not silently license an unsigned export. A
        # backup nobody can distinguish from a signed one is worse than no
        # backup, because it will be trusted.
        raise RuntimeError(
            "opti_oignon.pqc_signatures could not be imported: the signature "
            "posture cannot be determined, so an unsigned export cannot be "
            "distinguished from a signed one. Refusing."
        )

# Key name for storing PQC signature in backup dict
_PQC_SIGNATURE_KEY = "_pqc_signature"
_PQC_PUBLIC_KEY_KEY = "_pqc_public_key"

# Schema version for forward compatibility
BACKUP_SCHEMA_VERSION = "1.0"

# All known backup sections
BACKUP_SECTIONS = (
    "presets",
    "system_presets",
    "routing",
    "learned_routing",
    "plugins",
    "rag_metadata",
    "compression",
    "telemetry",
    "sandbox",
    "theme",
    "model_profiles",
    "cascading",
    "speculative",
    "benchmarks",
    "semantic_cache",
    "benchmark_auto_trigger",
    "humanizer",
    "fine_tune",
    "custom_pipelines",
    "execution_pipelines",
    "projects_settings",
)

# Valid import strategies
STRATEGY_MERGE = "merge"
STRATEGY_REPLACE = "replace"
VALID_STRATEGIES = (STRATEGY_MERGE, STRATEGY_REPLACE)


@dataclass
class BackupDiffItem:
    """A single difference found during import preview."""

    section: str = ""
    key: str = ""
    action: str = ""  # "add", "update", "skip" (merge only), "remove" (replace only)
    current_value: Any = None
    incoming_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d: dict[str, Any] = {
            "section": self.section,
            "key": self.key,
            "action": self.action,
        }
        if self.current_value is not None:
            d["current_value"] = self.current_value
        if self.incoming_value is not None:
            d["incoming_value"] = self.incoming_value
        return d


@dataclass
class BackupPreview:
    """Preview of what an import would change."""

    valid: bool = True
    strategy: str = STRATEGY_MERGE
    sections: list[str] = field(default_factory=list)
    diff: list[BackupDiffItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "valid": self.valid,
            "strategy": self.strategy,
            "sections": self.sections,
            "diff": [d.to_dict() for d in self.diff],
            "errors": self.errors,
            "summary": self.summary,
        }


@dataclass
class ImportResult:
    """Result of an import operation."""

    success: bool = False
    sections_imported: list[str] = field(default_factory=list)
    sections_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "success": self.success,
            "sections_imported": self.sections_imported,
            "sections_failed": self.sections_failed,
            "errors": self.errors,
            "rolled_back": self.rolled_back,
        }


class BackupManager:
    """Manages full configuration backup and restore.

    Collects configuration from all subsystems, serializes to JSON,
    and handles import with merge/replace strategies and rollback.
    """

    def __init__(self) -> None:
        self._section_exporters: dict[str, Any] = {}
        self._section_importers: dict[str, Any] = {}
        self._register_sections()

    # -----------------------------------------------------------------
    # Section registration
    # -----------------------------------------------------------------

    def _register_sections(self) -> None:
        """Register export/import handlers for each section."""
        self._section_exporters = {
            "presets": self._export_presets,
            "system_presets": self._export_system_presets,
            "routing": self._export_routing,
            "learned_routing": self._export_learned_routing,
            "plugins": self._export_plugins,
            "rag_metadata": self._export_rag_metadata,
            "compression": self._export_compression,
            "telemetry": self._export_telemetry,
            "sandbox": self._export_sandbox,
            "theme": self._export_theme,
            "model_profiles": self._export_model_profiles,
            "cascading": self._export_cascading,
            "speculative": self._export_speculative,
            "benchmarks": self._export_benchmarks,
            "semantic_cache": self._export_semantic_cache,
            "benchmark_auto_trigger": self._export_benchmark_auto_trigger,
            "humanizer": self._export_humanizer,
            "fine_tune": self._export_fine_tune,
            "custom_pipelines": self._export_custom_pipelines,
            "execution_pipelines": self._export_execution_pipelines,
            "projects_settings": self._export_projects_settings,
        }
        self._section_importers = {
            "presets": self._import_presets,
            "system_presets": self._import_system_presets,
            "routing": self._import_routing,
            "learned_routing": self._import_learned_routing,
            "plugins": self._import_plugins,
            "rag_metadata": self._import_rag_metadata,
            "compression": self._import_compression,
            "telemetry": self._import_telemetry,
            "sandbox": self._import_sandbox,
            "theme": self._import_theme,
            "model_profiles": self._import_model_profiles,
            "cascading": self._import_cascading,
            "speculative": self._import_speculative,
            "benchmarks": self._import_benchmarks,
            "semantic_cache": self._import_semantic_cache,
            "benchmark_auto_trigger": self._import_benchmark_auto_trigger,
            "humanizer": self._import_humanizer,
            "fine_tune": self._import_fine_tune,
            "custom_pipelines": self._import_custom_pipelines,
            "execution_pipelines": self._import_execution_pipelines,
            "projects_settings": self._import_projects_settings,
        }

    # -----------------------------------------------------------------
    # Public API -- Export
    # -----------------------------------------------------------------

    def export_all(self) -> dict[str, Any]:
        """Export all sections into a backup dict.

        Returns:
            Complete backup dict with metadata and all sections.
        """
        return self.export_sections(list(BACKUP_SECTIONS))

    def export_sections(self, sections: list[str]) -> dict[str, Any]:
        """Export specific sections into a backup dict.

        Args:
            sections: List of section names to include.

        Returns:
            Backup dict with metadata and requested sections.

        Raises:
            ValueError: If any section name is unknown.
        """
        unknown = [s for s in sections if s not in BACKUP_SECTIONS]
        if unknown:
            raise ValueError(f"Unknown backup sections: {', '.join(unknown)}")

        from opti_oignon.__version__ import __version__

        backup: dict[str, Any] = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "metadata": {
                "opti_oignon_version": __version__,
                "timestamp": time.time(),
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python_version": platform.python_version(),
                },
                "sections_included": list(sections),
            },
            "sections": {},
        }

        for section in sections:
            exporter = self._section_exporters.get(section)
            if exporter is None:
                logger.warning("No exporter for section: %s", section)
                continue
            try:
                data = exporter()
                backup["sections"][section] = data
                logger.debug("Exported section '%s': %d items", section, len(data) if isinstance(data, (dict, list)) else 1)
            except Exception as exc:
                logger.error("Failed to export section '%s': %s", section, exc)
                backup["sections"][section] = {"_error": str(exc)}

        # S129: PQC signature on exported backup
        self._sign_backup_pqc(backup)

        return backup

    # -----------------------------------------------------------------
    # Public API -- List sections
    # -----------------------------------------------------------------

    def list_sections(self) -> list[dict[str, Any]]:
        """List available backup sections with item counts.

        Returns:
            List of dicts with section name, description, and item count.
        """
        descriptions = {
            "presets": "User task presets",
            "system_presets": "System-level preset assignments",
            "routing": "Smart router configuration",
            "learned_routing": "Learned routing weights and config",
            "plugins": "Plugin states and configurations",
            "rag_metadata": "RAG collection and document metadata",
            "compression": "Conversation compressor settings",
            "telemetry": "Telemetry pipeline and history settings",
            "sandbox": "Sandbox isolation configuration",
            "theme": "User preferences and theme settings",
            "model_profiles": "Model profile assignments",
            "cascading": "Cascading inference tier config",
            "speculative": "Speculative decoding config",
            "benchmarks": "Custom benchmark profiles",
            "semantic_cache": "Semantic cache settings (config only, no cached content)",
            "benchmark_auto_trigger": "Benchmark auto-trigger settings",
            "humanizer": "Humanizer settings (config only, no feedback data)",
            "fine_tune": "Fine-tune settings and variants registry",
            "custom_pipelines": "User-authored agent pipelines",
            "execution_pipelines": "User-authored execution pipelines",
            "projects_settings": "Projects feature settings (no project content)",
        }

        result = []
        for section in BACKUP_SECTIONS:
            exporter = self._section_exporters.get(section)
            item_count = 0
            available = False
            if exporter:
                try:
                    data = exporter()
                    available = True
                    if isinstance(data, dict):
                        item_count = len(data)
                    elif isinstance(data, list):
                        item_count = len(data)
                    else:
                        item_count = 1
                except Exception:
                    available = False

            result.append({
                "name": section,
                "description": descriptions.get(section, ""),
                "item_count": item_count,
                "available": available,
            })

        return result

    # -----------------------------------------------------------------
    # Public API -- Validate
    # -----------------------------------------------------------------

    def validate_backup(self, data: dict[str, Any]) -> list[str]:
        """Validate a backup dict structure.

        Args:
            data: Backup dict to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        if not isinstance(data, dict):
            errors.append("Backup must be a JSON object")
            return errors

        # Schema version
        sv = data.get("schema_version")
        if sv is None:
            errors.append("Missing 'schema_version' field")
        elif not isinstance(sv, str):
            errors.append("'schema_version' must be a string")

        # Metadata
        meta = data.get("metadata")
        if meta is None:
            errors.append("Missing 'metadata' field")
        elif not isinstance(meta, dict):
            errors.append("'metadata' must be an object")

        # Sections
        sections = data.get("sections")
        if sections is None:
            errors.append("Missing 'sections' field")
        elif not isinstance(sections, dict):
            errors.append("'sections' must be an object")
        else:
            for key in sections:
                if key not in BACKUP_SECTIONS:
                    errors.append(f"Unknown section: '{key}'")

        return errors

    # -----------------------------------------------------------------
    # Public API -- Preview import
    # -----------------------------------------------------------------

    def preview_import(
        self,
        data: dict[str, Any],
        strategy: str = STRATEGY_MERGE,
    ) -> BackupPreview:
        """Preview what an import would change without applying.

        Args:
            data: Backup dict to preview.
            strategy: Import strategy ('merge' or 'replace').

        Returns:
            BackupPreview with diff items and summary.
        """
        preview = BackupPreview(strategy=strategy)

        if strategy not in VALID_STRATEGIES:
            preview.valid = False
            preview.errors.append(f"Invalid strategy: '{strategy}'. Must be one of: {', '.join(VALID_STRATEGIES)}")
            return preview

        validation_errors = self.validate_backup(data)
        if validation_errors:
            preview.valid = False
            preview.errors = validation_errors
            return preview

        sections = data.get("sections", {})
        preview.sections = list(sections.keys())

        counts: dict[str, int] = {"add": 0, "update": 0, "skip": 0}

        for section_name, section_data in sections.items():
            if section_name not in BACKUP_SECTIONS:
                continue
            if isinstance(section_data, dict) and "_error" in section_data:
                preview.errors.append(f"Section '{section_name}' has export error: {section_data['_error']}")
                continue

            try:
                current = self._export_section_safe(section_name)
                diff_items = self._compute_section_diff(
                    section_name, current, section_data, strategy
                )
                preview.diff.extend(diff_items)
                for item in diff_items:
                    counts[item.action] = counts.get(item.action, 0) + 1
            except Exception as exc:
                preview.errors.append(f"Preview error for '{section_name}': {str(exc)}")

        preview.summary = counts
        return preview

    # -----------------------------------------------------------------
    # Public API -- Import
    # -----------------------------------------------------------------

    def import_backup(
        self,
        data: dict[str, Any],
        strategy: str = STRATEGY_MERGE,
        allow_unsigned: bool = False,
    ) -> ImportResult:
        """Import a backup with the given strategy.

        Args:
            data: Backup dict to import.
            strategy: 'merge' (keep existing, add missing) or 'replace' (overwrite).
            allow_unsigned: BK-01. When a PQC keypair exists locally a valid
                signature is required by default; set True to explicitly
                override and restore an unsigned (or unverifiable) backup. A
                signature that is present but INVALID is always rejected and is
                never affected by this override.

        Returns:
            ImportResult with success/failure details.
        """
        result = ImportResult()

        if strategy not in VALID_STRATEGIES:
            result.errors.append(f"Invalid strategy: '{strategy}'")
            return result

        validation_errors = self.validate_backup(data)
        if validation_errors:
            result.errors = validation_errors
            return result

        # S129 / BK-01: PQC signature policy.
        pqc_check = self._verify_backup_pqc(data)
        if pqc_check is False:
            # Signature present but verification FAILED. Always rejected;
            # allow_unsigned never relaxes a failed verification.
            result.errors.append(
                "PQC signature verification failed. "
                "Backup may have been tampered with."
            )
            return result
        if pqc_check is None and not allow_unsigned:
            # BK-01: no verified signature. Distinguish the two None causes and
            # require a valid signature when this install has a PQC keypair.
            has_signature = bool(data.get(_PQC_SIGNATURE_KEY)) and bool(
                data.get(_PQC_PUBLIC_KEY_KEY)
            )
            if has_signature and not _PQC_LIB_AVAILABLE:
                result.errors.append(
                    "Backup carries a PQC signature but it cannot be verified "
                    "(liboqs/PQC unavailable). Refusing. Re-run with "
                    "allow_unsigned=True to override."
                )
                return result
            if not has_signature and pqc_keypair_exists():
                result.errors.append(
                    "Unsigned backup refused: a PQC keypair exists locally, so a "
                    "valid signature is required. Re-run with allow_unsigned=True "
                    "to override."
                )
                return result
            # Otherwise (no signature and no local keypair) PQC is not configured
            # here: allow the import for backward compatibility.
        # pqc_check is True (verified), or None accepted per the policy above.

        sections = data.get("sections", {})

        # Snapshot current state for rollback.
        # BK-05 (S194): a failed snapshot is recorded as a None sentinel,
        # never {} -- rolling back onto {} would wipe the live section
        # state (replace-mode importers treat {} as "delete everything").
        # _rollback already skips None snapshots with a warning.
        snapshots: dict[str, Any] = {}
        for section_name in sections:
            if section_name not in BACKUP_SECTIONS:
                continue
            exporter = self._section_exporters.get(section_name)
            if exporter is None:
                snapshots[section_name] = None
                continue
            try:
                snapshots[section_name] = exporter()
            except Exception as exc:
                logger.warning(
                    "Snapshot failed for '%s'; rollback for this section "
                    "will be skipped: %s",
                    section_name,
                    exc,
                )
                snapshots[section_name] = None

        # Apply each section
        applied: list[str] = []
        for section_name, section_data in sections.items():
            if section_name not in self._section_importers:
                result.errors.append(f"No importer for section: '{section_name}'")
                result.sections_failed.append(section_name)
                continue

            if isinstance(section_data, dict) and "_error" in section_data:
                result.errors.append(f"Skipping section '{section_name}' with export error")
                result.sections_failed.append(section_name)
                continue

            importer = self._section_importers[section_name]
            try:
                importer(section_data, strategy)
                applied.append(section_name)
                result.sections_imported.append(section_name)
                logger.info("Imported section '%s' with strategy '%s'", section_name, strategy)
            except Exception as exc:
                logger.error("Failed to import section '%s': %s", section_name, exc)
                result.errors.append(f"Import failed for '{section_name}': {str(exc)}")
                result.sections_failed.append(section_name)

                # Rollback all previously applied sections AND the failing
                # one: its importer may have partially applied before
                # raising (BK-04, S194). Its snapshot is in `snapshots`.
                self._rollback(applied + [section_name], snapshots)
                result.rolled_back = True
                return result

        result.success = len(result.sections_failed) == 0
        return result

    # -----------------------------------------------------------------
    # Rollback
    # -----------------------------------------------------------------

    def _rollback(self, applied: list[str], snapshots: dict[str, Any]) -> None:
        """Rollback previously applied sections to their snapshot state.

        Args:
            applied: List of section names that were successfully imported.
            snapshots: Dict of section_name -> snapshot data taken before import.
        """
        for section_name in reversed(applied):
            snapshot = snapshots.get(section_name)
            if snapshot is None:
                logger.warning("No snapshot for rollback of '%s'", section_name)
                continue
            importer = self._section_importers.get(section_name)
            if importer is None:
                continue
            try:
                importer(snapshot, STRATEGY_REPLACE)
                logger.info("Rolled back section '%s'", section_name)
            except Exception as exc:
                logger.error("Rollback failed for '%s': %s", section_name, exc)

    # -----------------------------------------------------------------
    # S129: PQC signing / verification helpers
    # -----------------------------------------------------------------

    def _sign_backup_pqc(self, backup: dict[str, Any]) -> None:
        """Sign a backup dict with PQC if enabled and keys are available.

        Modifies the backup dict in-place, adding '_pqc_signature' and
        '_pqc_public_key' keys at the top level.

        A broken promise is a REFUSAL, never a no-op. When post-quantum
        signing was required -- the operator asked for it, or the mode is a
        fortress, or the policy could not be read to tell -- and the primitive
        did not resolve, this raises rather than let a backup leave the machine
        unsigned while the caller believes it is signed. A symmetric MAC is not
        a weaker signature; it is a different security property, forgeable by
        whoever holds the shared secret.

        The gate is the REQUIREMENT, not the policy file. Asking the policy
        alone was the hole: with no signing block in it, the intent read False,
        this method returned, and a backup left a fortress host unsigned while
        every posture check upstream reported green -- because every one of them
        was asking whether the PRIMITIVE resolved, and it had. A fortress does
        not take instructions from a config file about its own trust root, any
        more than it does about its socket bind.

        When nothing was promised, it stays a no-op. A refusal there would be a
        denial of service the operator never asked for.
        """
        assert_pqc_posture()

        if not pqc_required():
            return
        if not pqc_keypair_exists():
            # The operator asked for signed backups, the primitive works, and
            # there is no key. That is a promise broken by absence rather than
            # by breakage, and it was a debug line.
            raise RuntimeError(
                "Post-quantum backup signatures are enabled and no keypair "
                "exists. Refusing to export a backup that would silently carry "
                "no signature. Generate a keypair, or turn the setting off "
                "deliberately."
            )

        try:
            public_key, private_key = load_pqc_keypair()

            # Serialize the backup WITHOUT any existing PQC fields
            clean = {k: v for k, v in backup.items()
                     if k not in (_PQC_SIGNATURE_KEY, _PQC_PUBLIC_KEY_KEY)}
            payload = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")

            signature = _pqc_sign(payload, private_key)

            backup[_PQC_SIGNATURE_KEY] = base64.urlsafe_b64encode(signature).decode("ascii")
            backup[_PQC_PUBLIC_KEY_KEY] = base64.urlsafe_b64encode(public_key).decode("ascii")
            logger.info("PQC signature applied to backup (%d bytes)", len(signature))
        except Exception as exc:
            # The asymmetry this replaces was indefensible. A promise broken by
            # ABSENCE -- no keypair -- refused, loudly, above. A promise broken
            # by BREAKAGE -- the key will not load, the backend rejects it, the
            # signer dies mid-flight -- logged a warning that called the result
            # valid, and shipped the document unsigned. The second is the more
            # dangerous of the two: a missing key is discoverable, a swallowed
            # exception is not. Signing was REQUIRED by the time we got here.
            raise RuntimeError(
                "Post-quantum backup signing was required and failed: "
                f"{exc}. Refusing to export a backup the caller would believe "
                "is signed. A document nobody can distinguish from a signed one "
                "is worse than no document, because it will be trusted."
            ) from exc

    def _verify_backup_pqc(self, data: dict[str, Any]) -> bool | None:
        """Verify PQC signature on an imported backup.

        Returns:
            True:  Signature present and verified.
            False: Signature present but verification FAILED.
            None:  No signature present, or PQC library unavailable
                   (backward compat: allow import with warning).
        """
        sig_b64 = data.get(_PQC_SIGNATURE_KEY)
        pub_b64 = data.get(_PQC_PUBLIC_KEY_KEY)

        if not sig_b64 or not pub_b64:
            # No PQC signature in this backup -- allow import
            return None

        if not _PQC_LIB_AVAILABLE:
            logger.warning(
                "Backup has PQC signature but liboqs is not installed. "
                "Signature cannot be verified. Allowing import anyway."
            )
            return None

        try:
            signature = base64.urlsafe_b64decode(sig_b64)
            public_key = base64.urlsafe_b64decode(pub_b64)

            # Reconstruct the signed payload (without PQC fields)
            clean = {k: v for k, v in data.items()
                     if k not in (_PQC_SIGNATURE_KEY, _PQC_PUBLIC_KEY_KEY)}
            payload = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")

            is_valid = _pqc_verify(payload, signature, public_key)
            if is_valid:
                logger.info("PQC signature on backup verified successfully")
                return True
            else:
                logger.error("PQC signature verification FAILED on backup")
                return False
        except Exception as exc:
            logger.error("PQC verification error: %s", exc)
            return False

    # -----------------------------------------------------------------
    # Diff computation
    # -----------------------------------------------------------------

    def _compute_section_diff(
        self,
        section: str,
        current: Any,
        incoming: Any,
        strategy: str,
    ) -> list[BackupDiffItem]:
        """Compute diff items for a section.

        For dict-type sections, compares keys.
        For other types, compares the whole value.
        """
        items: list[BackupDiffItem] = []

        if isinstance(current, dict) and isinstance(incoming, dict):
            all_keys = set(list(current.keys()) + list(incoming.keys()))
            for key in sorted(all_keys):
                in_current = key in current
                in_incoming = key in incoming

                if in_incoming and not in_current:
                    items.append(BackupDiffItem(
                        section=section, key=key, action="add",
                        incoming_value=_summarize(incoming[key]),
                    ))
                elif in_incoming and in_current:
                    if current[key] != incoming[key]:
                        if strategy == STRATEGY_MERGE:
                            items.append(BackupDiffItem(
                                section=section, key=key, action="skip",
                                current_value=_summarize(current[key]),
                                incoming_value=_summarize(incoming[key]),
                            ))
                        else:
                            items.append(BackupDiffItem(
                                section=section, key=key, action="update",
                                current_value=_summarize(current[key]),
                                incoming_value=_summarize(incoming[key]),
                            ))
                # key only in current: nothing to do for either strategy
        else:
            # Non-dict section: treat as single value
            if current != incoming:
                action = "skip" if strategy == STRATEGY_MERGE and current else "update"
                if not current and incoming:
                    action = "add"
                items.append(BackupDiffItem(
                    section=section, key="_all",
                    action=action,
                    current_value=_summarize(current),
                    incoming_value=_summarize(incoming),
                ))

        return items

    def _export_section_safe(self, section_name: str) -> Any:
        """Export a section, returning empty dict on failure."""
        exporter = self._section_exporters.get(section_name)
        if exporter is None:
            return {}
        try:
            return exporter()
        except Exception:
            return {}

    # =================================================================
    # Section exporters
    # =================================================================

    def _export_presets(self) -> dict[str, Any]:
        """Export user presets."""
        try:
            from opti_oignon.presets import preset_manager
            all_presets = preset_manager.get_all()
            return {pid: p.to_dict() for pid, p in all_presets.items()}
        except Exception as exc:
            logger.debug("Cannot export presets: %s", exc)
            return {}

    def _export_system_presets(self) -> dict[str, Any]:
        """Export system preset assignments."""
        try:
            from opti_oignon.system_presets import system_presets_manager
            if system_presets_manager is None:
                return {}
            presets = system_presets_manager.list_presets()
            return {p.name: p.to_dict() for p in presets}
        except Exception as exc:
            logger.debug("Cannot export system presets: %s", exc)
            return {}

    def _export_routing(self) -> dict[str, Any]:
        """Export smart router configuration."""
        try:
            from opti_oignon.smart_router import smart_router
            if smart_router is None:
                return {}
            return smart_router.get_config()
        except Exception as exc:
            logger.debug("Cannot export routing: %s", exc)
            return {}

    def _export_learned_routing(self) -> dict[str, Any]:
        """Export learned router config and weights."""
        try:
            from opti_oignon.learned_router import learned_router
            if learned_router is None:
                return {}
            config = learned_router.get_config()
            try:
                weights = learned_router.to_dict()
                config["_weights"] = weights
            except Exception:
                pass
            return config
        except Exception as exc:
            logger.debug("Cannot export learned routing: %s", exc)
            return {}

    def _export_plugins(self) -> dict[str, Any]:
        """Export plugin states and configs."""
        try:
            from opti_oignon.plugin_manifest import plugin_registry
            if plugin_registry is None:
                return {}
            plugins = plugin_registry.list_plugins()
            result = {}
            for record in plugins:
                result[record.manifest.name] = {
                    "state": record.state,
                    "config": record.config,
                    "manifest": record.manifest.to_dict(),
                }
            return result
        except Exception as exc:
            logger.debug("Cannot export plugins: %s", exc)
            return {}

    def _export_rag_metadata(self) -> dict[str, Any]:
        """Export RAG collection metadata (NOT vectors)."""
        try:
            from opti_oignon.rag_store import rag_store
            if rag_store is None:
                return {}
            collections = rag_store.list_collections()
            result = {}
            for col in collections:
                if isinstance(col, dict):
                    name = col.get("name", "")
                    result[name] = col
                else:
                    result[col.name] = col.to_dict()
            return result
        except Exception as exc:
            logger.debug("Cannot export RAG metadata: %s", exc)
            return {}

    def _export_compression(self) -> dict[str, Any]:
        """Export conversation compressor settings."""
        try:
            from opti_oignon.conversation_compressor import conversation_compressor
            if conversation_compressor is None:
                return {}
            return conversation_compressor.get_config()
        except Exception as exc:
            logger.debug("Cannot export compression: %s", exc)
            return {}

    def _export_telemetry(self) -> dict[str, Any]:
        """Export telemetry settings."""
        result: dict[str, Any] = {}
        # Pipeline config from YAML
        try:
            cfg_path = Path(__file__).parent / "config" / "telemetry.yaml"
            if cfg_path.is_file():
                import yaml
                with open(cfg_path, encoding="utf-8") as f:
                    result["pipeline"] = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.debug("Cannot export telemetry pipeline config: %s", exc)

        # History settings
        try:
            from opti_oignon.telemetry_history import telemetry_history_store
            if telemetry_history_store is not None:
                result["history"] = {
                    "retention_days": telemetry_history_store._retention_days,
                    "auto_purge_enabled": telemetry_history_store._auto_purge_enabled,
                }
        except Exception as exc:
            logger.debug("Cannot export telemetry history settings: %s", exc)

        return result

    def _export_sandbox(self) -> dict[str, Any]:
        """Export sandbox configuration from YAML."""
        try:
            cfg_path = Path(__file__).parent / "config" / "sandbox.yaml"
            if cfg_path.is_file():
                import yaml
                with open(cfg_path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.debug("Cannot export sandbox config: %s", exc)
        return {}

    def _export_theme(self) -> dict[str, Any]:
        """Export user preferences and theme settings."""
        try:
            from opti_oignon.config import config
            if config is None:
                return {}
            data = config.as_dict()
            return data.get("user", {})
        except Exception as exc:
            logger.debug("Cannot export theme: %s", exc)
            return {}

    def _export_model_profiles(self) -> dict[str, Any]:
        """Export model profile assignments."""
        try:
            from opti_oignon.model_profiles import profile_manager
            if profile_manager is None:
                return {}
            profiles = profile_manager.list_profiles()
            return {p.name: p.to_dict() for p in profiles}
        except Exception as exc:
            logger.debug("Cannot export model profiles: %s", exc)
            return {}

    def _export_cascading(self) -> dict[str, Any]:
        """Export cascading inference configuration."""
        try:
            from opti_oignon.cascading import cascading_inference
            if cascading_inference is None:
                return {}
            return cascading_inference.get_config()
        except Exception as exc:
            logger.debug("Cannot export cascading: %s", exc)
            return {}

    def _export_speculative(self) -> dict[str, Any]:
        """Export speculative decoding configuration."""
        try:
            from opti_oignon.speculative_decoding import get_speculative_decoding_manager
            if get_speculative_decoding_manager is None:
                return {}
            mgr = get_speculative_decoding_manager()
            if mgr is None:
                return {}
            return {
                "config": mgr._config.to_dict(),
                "family_compatibility": dict(mgr._family_compat),
                "vram_budget": dict(mgr._vram_budget_cfg),
            }
        except Exception as exc:
            logger.debug("Cannot export speculative: %s", exc)
            return {}

    def _export_benchmarks(self) -> dict[str, Any]:
        """Export custom benchmark profiles."""
        try:
            from opti_oignon.benchmark_custom_profiles import custom_profile_store
            if custom_profile_store is None:
                return {}
            profiles = custom_profile_store.list_profiles()
            return {p.profile_id: p.to_dict() for p in profiles}
        except Exception as exc:
            logger.debug("Cannot export benchmarks: %s", exc)
            return {}

    def _export_semantic_cache(self) -> dict[str, Any]:
        """Export semantic cache configuration (S220 BK-06).

        Config only: cached entries are regenerable content and stay
        excluded by design (ATREST_INVENTORY).
        """
        try:
            from opti_oignon.semantic_cache import semantic_cache
            if semantic_cache is None:
                return {}
            return semantic_cache.get_config()
        except Exception as exc:
            logger.debug("Cannot export semantic cache config: %s", exc)
            return {}

    def _export_benchmark_auto_trigger(self) -> dict[str, Any]:
        """Export benchmark auto-trigger configuration (S220 BK-06).

        Custom benchmark profiles already live in the 'benchmarks'
        section; this covers the auto-trigger settings only.
        """
        try:
            from opti_oignon.benchmark_auto_trigger import auto_trigger
            if auto_trigger is None:
                return {}
            return dict(auto_trigger.config)
        except Exception as exc:
            logger.debug("Cannot export benchmark auto-trigger config: %s", exc)
            return {}

    def _export_humanizer(self) -> dict[str, Any]:
        """Export humanizer configuration (S220 BK-06).

        Config only: the humanizer feedback store is data and stays
        excluded by design (ATREST_INVENTORY).
        """
        try:
            from opti_oignon.humanizer import humanizer_engine
            if humanizer_engine is None:
                return {}
            return humanizer_engine.get_config()
        except Exception as exc:
            logger.debug("Cannot export humanizer config: %s", exc)
            return {}

    def _export_fine_tune(self) -> dict[str, Any]:
        """Export fine-tune config and the variants registry (S220 BK-06).

        Two halves: 'config' (config/fine_tune.yaml) and 'variants'
        (the user-authored registry). A/B comparison results are
        telemetry-class data and stay excluded by design.
        """
        result: dict[str, Any] = {}
        try:
            cfg_path = Path(__file__).parent / "config" / "fine_tune.yaml"
            if cfg_path.is_file():
                import yaml
                with open(cfg_path, encoding="utf-8") as f:
                    result["config"] = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.debug("Cannot export fine-tune config: %s", exc)
        try:
            from opti_oignon.fine_tune_tracker import fine_tune_tracker
            if fine_tune_tracker is not None:
                variants = fine_tune_tracker.list_variants()
                result["variants"] = {v.variant_id: v.to_dict() for v in variants}
        except Exception as exc:
            logger.debug("Cannot export fine-tune variants: %s", exc)
        return result

    def _export_custom_pipelines(self) -> dict[str, Any]:
        """Export user-authored agent pipelines (S220 BK-06).

        Data section: data/pipelines_custom.yaml is user-authored.
        Builtin pipelines ship with the install and are excluded.
        """
        try:
            from opti_oignon.pipeline_manager import get_pipeline_manager
            mgr = get_pipeline_manager()
            if mgr is None:
                return {}
            return {p.id: p.to_dict() for p in mgr.list_custom()}
        except Exception as exc:
            logger.debug("Cannot export custom pipelines: %s", exc)
            return {}

    def _export_execution_pipelines(self) -> dict[str, Any]:
        """Export user-authored execution pipelines (S220 BK-06).

        Data section: data/execution_pipelines.yaml is user-authored.
        Builtin pipelines ship with the install and are excluded.
        """
        try:
            from opti_oignon.pipelines import get_pipeline_store
            store = get_pipeline_store()
            if store is None:
                return {}
            return {p.id: p.to_dict() for p in store.list_custom()}
        except Exception as exc:
            logger.debug("Cannot export execution pipelines: %s", exc)
            return {}

    def _export_projects_settings(self) -> dict[str, Any]:
        """Export projects feature settings from YAML (S220 BK-06).

        Settings only (config/projects.yaml): project content (the
        projects DB and project files) is never in the backup.
        """
        try:
            cfg_path = Path(__file__).parent / "config" / "projects.yaml"
            if cfg_path.is_file():
                import yaml
                with open(cfg_path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.debug("Cannot export projects settings: %s", exc)
        return {}

    # =================================================================
    # Section importers
    # =================================================================

    def _import_presets(self, data: dict[str, Any], strategy: str) -> None:
        """Import user presets."""
        from opti_oignon.presets import preset_manager
        if strategy == STRATEGY_REPLACE:
            # Clear existing and add all from backup
            for pid in list(preset_manager.get_all().keys()):
                try:
                    preset_manager.delete(pid)
                except Exception:
                    pass
        for pid, pdata in data.items():
            existing = preset_manager.get(pid)
            if existing and strategy == STRATEGY_MERGE:
                continue  # Keep existing
            try:
                preset_manager.create_from_dict(pid, pdata)
            except Exception:
                try:
                    preset_manager.update_from_dict(pid, pdata)
                except Exception as exc:
                    logger.warning("Cannot import preset '%s': %s", pid, exc)

    def _import_system_presets(self, data: dict[str, Any], strategy: str) -> None:
        """Import system preset assignments."""
        # System presets are read-only definitions; we import applied state
        from opti_oignon.system_presets import system_presets_manager
        if system_presets_manager is None:
            raise RuntimeError("System presets module not available")
        # System presets are typically not user-modifiable, skip silently
        logger.debug("System presets import: noted (read-only definitions)")

    def _import_routing(self, data: dict[str, Any], strategy: str) -> None:
        """Import smart router configuration."""
        from opti_oignon.smart_router import smart_router
        if smart_router is None:
            raise RuntimeError("Smart router not available")
        update_kwargs: dict[str, Any] = {}
        if "enabled" in data:
            update_kwargs["enabled"] = data["enabled"]
        if "default_model" in data:
            update_kwargs["default_model"] = data["default_model"]
        if "speed_preference" in data:
            update_kwargs["speed_preference"] = data["speed_preference"]
        if "speed_weights" in data:
            update_kwargs["speed_weights"] = data["speed_weights"]
        if update_kwargs:
            smart_router.update_config(**update_kwargs)

    def _import_learned_routing(self, data: dict[str, Any], strategy: str) -> None:
        """Import learned routing config."""
        from opti_oignon.learned_router import learned_router
        if learned_router is None:
            raise RuntimeError("Learned router not available")
        # Extract weights if present
        config_data = {k: v for k, v in data.items() if k != "_weights"}
        if config_data:
            if strategy == STRATEGY_REPLACE:
                learned_router.update_config(config_data)
            else:
                # Merge: only add keys not present in current config
                current = learned_router.get_config()
                updates = {k: v for k, v in config_data.items() if k not in current}
                if updates:
                    learned_router.update_config(updates)

    def _import_plugins(self, data: dict[str, Any], strategy: str) -> None:
        """Import plugin states and configs."""
        from opti_oignon.plugin_manifest import plugin_registry
        if plugin_registry is None:
            raise RuntimeError("Plugin registry not available")
        for name, pdata in data.items():
            try:
                state = pdata.get("state", "")
                config = pdata.get("config", {})
                existing = None
                for rec in plugin_registry.list_plugins():
                    if rec.manifest.name == name:
                        existing = rec
                        break
                if existing is None:
                    continue  # Cannot install new plugins via backup
                if strategy == STRATEGY_MERGE and existing.state:
                    # Keep current state in merge mode
                    pass
                else:
                    if state:
                        plugin_registry.set_state(name, state)
                if config:
                    if strategy == STRATEGY_REPLACE:
                        plugin_registry.set_config(name, config)
                    else:
                        # Merge: add missing config keys
                        merged = dict(existing.config)
                        for k, v in config.items():
                            if k not in merged:
                                merged[k] = v
                        plugin_registry.set_config(name, merged)
            except Exception as exc:
                logger.warning("Cannot import plugin '%s': %s", name, exc)

    def _import_rag_metadata(self, data: dict[str, Any], strategy: str) -> None:
        """Import RAG collection metadata (creates missing collections)."""
        try:
            from opti_oignon.rag_store import rag_store
            if rag_store is None:
                raise RuntimeError("RAG store not available")
            existing_names = set()
            for col in rag_store.list_collections():
                if isinstance(col, dict):
                    existing_names.add(col.get("name", ""))
                else:
                    existing_names.add(col.name)
            for name, col_data in data.items():
                if name not in existing_names:
                    try:
                        rag_store.create_collection(name)
                        logger.info("Created RAG collection '%s' from backup", name)
                    except Exception as exc:
                        logger.warning("Cannot create collection '%s': %s", name, exc)
        except Exception as exc:
            logger.debug("RAG metadata import skipped: %s", exc)

    def _import_compression(self, data: dict[str, Any], strategy: str) -> None:
        """Import conversation compressor settings."""
        from opti_oignon.conversation_compressor import conversation_compressor
        if conversation_compressor is None:
            raise RuntimeError("Conversation compressor not available")
        if strategy == STRATEGY_REPLACE:
            conversation_compressor.update_config(data)
        else:
            current = conversation_compressor.get_config()
            updates = {k: v for k, v in data.items() if k not in current}
            if updates:
                conversation_compressor.update_config(updates)

    def _import_telemetry(self, data: dict[str, Any], strategy: str) -> None:
        """Import telemetry settings."""
        # Pipeline config -> write YAML
        pipeline = data.get("pipeline")
        if pipeline and isinstance(pipeline, dict):
            try:
                import yaml
                cfg_path = Path(__file__).parent / "config" / "telemetry.yaml"
                if strategy == STRATEGY_REPLACE:
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(pipeline, f, default_flow_style=False)
                elif strategy == STRATEGY_MERGE and cfg_path.is_file():
                    with open(cfg_path, encoding="utf-8") as f:
                        current = yaml.safe_load(f) or {}
                    merged = dict(current)
                    for k, v in pipeline.items():
                        if k not in merged:
                            merged[k] = v
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(merged, f, default_flow_style=False)
            except Exception as exc:
                logger.warning("Cannot import telemetry pipeline config: %s", exc)

        # History settings
        history = data.get("history")
        if history and isinstance(history, dict):
            try:
                from opti_oignon.telemetry_history import telemetry_history_store
                if telemetry_history_store is not None:
                    telemetry_history_store.update_settings(
                        retention_days=history.get("retention_days"),
                        auto_purge_enabled=history.get("auto_purge_enabled"),
                    )
            except Exception as exc:
                logger.warning("Cannot import telemetry history settings: %s", exc)

    def _import_sandbox(self, data: dict[str, Any], strategy: str) -> None:
        """Import sandbox configuration."""
        try:
            import yaml
            cfg_path = Path(__file__).parent / "config" / "sandbox.yaml"
            if strategy == STRATEGY_REPLACE:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False)
            elif strategy == STRATEGY_MERGE and cfg_path.is_file():
                with open(cfg_path, encoding="utf-8") as f:
                    current = yaml.safe_load(f) or {}
                merged = dict(current)
                for k, v in data.items():
                    if k not in merged:
                        merged[k] = v
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged, f, default_flow_style=False)
        except Exception as exc:
            logger.warning("Cannot import sandbox config: %s", exc)

    def _import_theme(self, data: dict[str, Any], strategy: str) -> None:
        """Import user preferences and theme settings."""
        from opti_oignon.config import config
        if config is None:
            raise RuntimeError("Config module not available")
        for key, value in data.items():
            if strategy == STRATEGY_MERGE:
                existing = config.get_user_preference(key)
                if existing is not None:
                    continue  # Keep existing
            config.set_user_preference(key, value)

    def _import_model_profiles(self, data: dict[str, Any], strategy: str) -> None:
        """Import model profile assignments."""
        from opti_oignon.model_profiles import profile_manager
        if profile_manager is None:
            raise RuntimeError("Model profiles not available")
        existing_names = {p.name for p in profile_manager.list_profiles()}
        for name, pdata in data.items():
            if strategy == STRATEGY_MERGE and name in existing_names:
                continue  # Keep existing
            try:
                if name in existing_names:
                    profile_manager.update_profile(name, pdata)
                else:
                    profile_manager.create_profile(pdata)
            except Exception as exc:
                logger.warning("Cannot import model profile '%s': %s", name, exc)

    def _import_cascading(self, data: dict[str, Any], strategy: str) -> None:
        """Import cascading inference configuration."""
        from opti_oignon.cascading import cascading_inference
        if cascading_inference is None:
            raise RuntimeError("Cascading inference not available")
        update_kwargs: dict[str, Any] = {}
        if "enabled" in data:
            update_kwargs["enabled"] = data["enabled"]
        if "tiers" in data:
            update_kwargs["tiers"] = data["tiers"]
        if "timeout_per_tier" in data:
            update_kwargs["timeout_per_tier_seconds"] = data["timeout_per_tier"]
        if "score_weights" in data:
            update_kwargs["score_weights"] = data["score_weights"]
        if update_kwargs:
            cascading_inference.update_config(**update_kwargs)

    def _import_speculative(self, data: dict[str, Any], strategy: str) -> None:
        """Import speculative decoding configuration."""
        try:
            import yaml
            cfg_path = Path(__file__).parent / "config" / "speculative_decoding.yaml"
            spec_config = data.get("config", {})
            if spec_config:
                backup_yaml: dict[str, Any] = {"speculative_decoding": spec_config}
                if "family_compatibility" in data:
                    backup_yaml["family_compatibility"] = data["family_compatibility"]
                if "vram_budget" in data:
                    backup_yaml["vram_budget"] = data["vram_budget"]

                if strategy == STRATEGY_REPLACE:
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(backup_yaml, f, default_flow_style=False)
                elif strategy == STRATEGY_MERGE and cfg_path.is_file():
                    with open(cfg_path, encoding="utf-8") as f:
                        current = yaml.safe_load(f) or {}
                    for k, v in backup_yaml.items():
                        if k not in current:
                            current[k] = v
                        elif isinstance(current[k], dict) and isinstance(v, dict):
                            for sk, sv in v.items():
                                if sk not in current[k]:
                                    current[k][sk] = sv
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(current, f, default_flow_style=False)
        except Exception as exc:
            logger.warning("Cannot import speculative config: %s", exc)

    def _import_benchmarks(self, data: dict[str, Any], strategy: str) -> None:
        """Import custom benchmark profiles."""
        from opti_oignon.benchmark_custom_profiles import custom_profile_store
        if custom_profile_store is None:
            raise RuntimeError("Custom profile store not available")
        existing_ids = {p.profile_id for p in custom_profile_store.list_profiles()}
        for pid, pdata in data.items():
            if strategy == STRATEGY_MERGE and pid in existing_ids:
                continue
            try:
                if pid in existing_ids:
                    custom_profile_store.update_profile(pid, pdata)
                else:
                    custom_profile_store.create_profile(pdata)
            except Exception as exc:
                logger.warning("Cannot import benchmark profile '%s': %s", pid, exc)

    def _import_semantic_cache(self, data: dict[str, Any], strategy: str) -> None:
        """Import semantic cache configuration (config only)."""
        from opti_oignon.semantic_cache import semantic_cache
        if semantic_cache is None:
            raise RuntimeError("Semantic cache not available")
        if not isinstance(data, dict) or not data:
            return
        if strategy == STRATEGY_MERGE:
            current = semantic_cache.get_config()
            updates = {k: v for k, v in data.items() if k not in current}
        else:
            updates = dict(data)
        if updates:
            semantic_cache.update_config(updates)

    def _import_benchmark_auto_trigger(self, data: dict[str, Any], strategy: str) -> None:
        """Import benchmark auto-trigger configuration."""
        from opti_oignon.benchmark_auto_trigger import auto_trigger
        if auto_trigger is None:
            raise RuntimeError("Benchmark auto-trigger not available")
        if not isinstance(data, dict) or not data:
            return
        if strategy == STRATEGY_MERGE:
            current = dict(auto_trigger.config)
            updates = {k: v for k, v in data.items() if k not in current}
        else:
            updates = dict(data)
        if updates:
            auto_trigger.update_config(updates)

    def _import_humanizer(self, data: dict[str, Any], strategy: str) -> None:
        """Import humanizer configuration (config only)."""
        from opti_oignon.humanizer import humanizer_engine
        if humanizer_engine is None:
            raise RuntimeError("Humanizer not available")
        if not isinstance(data, dict) or not data:
            return
        if strategy == STRATEGY_MERGE:
            current = humanizer_engine.get_config()
            updates = {k: v for k, v in data.items() if k not in current}
        else:
            updates = dict(data)
        if updates:
            humanizer_engine.update_config(**updates)

    def _import_fine_tune(self, data: dict[str, Any], strategy: str) -> None:
        """Import fine-tune config and the variants registry.

        The registry has no delete API, so 'replace' is an upsert
        (update existing ids, register missing ones); variants absent
        from the backup are never cleared.
        """
        cfg = data.get("config")
        if cfg and isinstance(cfg, dict):
            try:
                import yaml
                cfg_path = Path(__file__).parent / "config" / "fine_tune.yaml"
                if strategy == STRATEGY_REPLACE:
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg, f, default_flow_style=False)
                elif strategy == STRATEGY_MERGE and cfg_path.is_file():
                    with open(cfg_path, encoding="utf-8") as f:
                        current = yaml.safe_load(f) or {}
                    merged = dict(current)
                    for k, v in cfg.items():
                        if k not in merged:
                            merged[k] = v
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(merged, f, default_flow_style=False)
            except Exception as exc:
                logger.warning("Cannot import fine-tune config: %s", exc)

        variants = data.get("variants")
        if variants and isinstance(variants, dict):
            from opti_oignon.fine_tune_tracker import (
                FineTuneVariant,
                fine_tune_tracker,
            )
            if fine_tune_tracker is None:
                raise RuntimeError("Fine-tune tracker not available")
            existing_ids = {v.variant_id for v in fine_tune_tracker.list_variants()}
            for vid, vdata in variants.items():
                if not isinstance(vdata, dict):
                    continue
                try:
                    if vid in existing_ids:
                        if strategy == STRATEGY_MERGE:
                            continue
                        fine_tune_tracker.update_variant(vid, vdata)
                    else:
                        fine_tune_tracker.register_variant(
                            FineTuneVariant.from_dict(vdata)
                        )
                except Exception as exc:
                    logger.warning(
                        "Cannot import fine-tune variant '%s': %s", vid, exc
                    )

    def _import_custom_pipelines(self, data: dict[str, Any], strategy: str) -> None:
        """Import user-authored agent pipelines.

        Replace clears existing CUSTOM pipelines first; builtin
        pipelines are never touched (delete refuses builtins).
        """
        from opti_oignon.pipeline_manager import Pipeline, get_pipeline_manager
        mgr = get_pipeline_manager()
        if mgr is None:
            raise RuntimeError("Pipeline manager not available")
        if strategy == STRATEGY_REPLACE:
            for p in list(mgr.list_custom()):
                try:
                    mgr.delete(p.id)
                except Exception:
                    pass
        existing_ids = {p.id for p in mgr.list_custom()}
        for pid, pdata in data.items():
            if not isinstance(pdata, dict):
                continue
            if strategy == STRATEGY_MERGE and pid in existing_ids:
                continue
            try:
                pipeline = Pipeline.from_dict(pid, pdata, is_builtin=False)
                if pid in existing_ids:
                    mgr.update(pid, pipeline)
                else:
                    mgr.create(pipeline)
            except Exception as exc:
                logger.warning("Cannot import custom pipeline '%s': %s", pid, exc)

    def _import_execution_pipelines(self, data: dict[str, Any], strategy: str) -> None:
        """Import user-authored execution pipelines.

        Replace clears existing CUSTOM pipelines first; builtin
        pipelines are never touched (delete refuses builtins).
        """
        from opti_oignon.pipelines import ExecutionPipeline, get_pipeline_store
        store = get_pipeline_store()
        if store is None:
            raise RuntimeError("Pipeline store not available")
        if strategy == STRATEGY_REPLACE:
            for p in list(store.list_custom()):
                try:
                    store.delete(p.id)
                except Exception:
                    pass
        existing_ids = {p.id for p in store.list_custom()}
        for pid, pdata in data.items():
            if not isinstance(pdata, dict):
                continue
            if strategy == STRATEGY_MERGE and pid in existing_ids:
                continue
            try:
                pipeline = ExecutionPipeline.from_dict(pid, pdata, is_builtin=False)
                if pid in existing_ids:
                    store.update(pid, pipeline)
                else:
                    store.create(pipeline)
            except Exception as exc:
                logger.warning(
                    "Cannot import execution pipeline '%s': %s", pid, exc
                )

    def _import_projects_settings(self, data: dict[str, Any], strategy: str) -> None:
        """Import projects feature settings (settings only, YAML)."""
        if not isinstance(data, dict) or not data:
            return
        try:
            import yaml
            cfg_path = Path(__file__).parent / "config" / "projects.yaml"
            if strategy == STRATEGY_REPLACE:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False)
            elif strategy == STRATEGY_MERGE and cfg_path.is_file():
                with open(cfg_path, encoding="utf-8") as f:
                    current = yaml.safe_load(f) or {}
                merged = dict(current)
                for k, v in data.items():
                    if k not in merged:
                        merged[k] = v
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged, f, default_flow_style=False)
        except Exception as exc:
            logger.warning("Cannot import projects settings: %s", exc)


# =====================================================================
# Helpers
# =====================================================================

def _summarize(value: Any) -> Any:
    """Create a brief summary of a value for diff display.

    Truncates large strings and dicts to keep diffs readable.
    """
    if isinstance(value, str) and len(value) > 100:
        return value[:97] + "..."
    if isinstance(value, dict) and len(value) > 5:
        keys = list(value.keys())[:5]
        return {k: value[k] for k in keys}
    if isinstance(value, list) and len(value) > 5:
        return value[:5]
    return value


# =====================================================================
# S125: Encrypted backup support
# =====================================================================

# Encrypted backup magic bytes for format detection
_ENCRYPTED_MAGIC = b"OOENC1"


def _load_backup_config() -> dict:
    """Load backup encryption config from security.yaml (S125)."""
    import yaml
    cfg_path = Path(__file__).parent / "config" / "security.yaml"
    try:
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("backup", {})
    except Exception:
        pass
    return {}


def encrypt_backup(backup_data: dict[str, Any], password: str) -> bytes:
    """Encrypt a backup dict with a password (S125, hardened).

    Uses Argon2id (or PBKDF2 fallback) key derivation and AES-256-GCM.

    Format: OOENC1 || kdf_id(2) || salt(16) || encrypted_data(variable)

    Args:
        backup_data: The backup dict (as returned by export_all).
        password: User-provided password for encryption.

    Returns:
        Encrypted bytes ready to write to a .oo-backup.enc file.

    Raises:
        ValueError: If password is too short.
    """
    if not password or len(password) < 8:
        raise ValueError("Backup encryption password must be at least 8 characters")

    try:
        from .encryption import (
            _KDF_ARGON2ID,
            _KDF_PBKDF2,
            derive_key_from_passphrase,
            encrypt_bytes,
        )
    except ImportError:
        raise RuntimeError("Encryption module not available")

    # Serialize backup to JSON
    plaintext = json.dumps(backup_data, ensure_ascii=False, indent=2).encode("utf-8")

    # Derive key from password
    key, salt, kdf_name = derive_key_from_passphrase(password)

    # KDF identifier for decryption
    kdf_id = _KDF_ARGON2ID if kdf_name == "argon2id" else _KDF_PBKDF2

    # Encrypt with AES-256-GCM
    encrypted = encrypt_bytes(key, plaintext)

    # Build encrypted file: magic + kdf_id + salt + encrypted
    return _ENCRYPTED_MAGIC + kdf_id + salt + encrypted


def decrypt_backup(encrypted_data: bytes, password: str) -> dict[str, Any]:
    """Decrypt an encrypted backup file (S125, hardened).

    Args:
        encrypted_data: Raw bytes from an .oo-backup.enc file.
        password: Password used to encrypt the backup.

    Returns:
        The decrypted backup dict.

    Raises:
        ValueError: If magic bytes don't match, password is wrong, or data is corrupt.
    """
    magic_len = len(_ENCRYPTED_MAGIC)
    min_size = magic_len + 2 + 16 + 29  # magic + kdf_id + salt + min encrypted
    if not encrypted_data or len(encrypted_data) < min_size:
        raise ValueError("Encrypted backup data is too short or empty")

    magic = encrypted_data[:magic_len]
    if magic != _ENCRYPTED_MAGIC:
        raise ValueError("Not an encrypted Opti-Oignon backup (invalid magic bytes)")

    try:
        from .encryption import _KDF_ARGON2ID, decrypt_bytes, derive_key_from_passphrase
    except ImportError:
        raise RuntimeError("Encryption module not available")

    # Extract kdf_id, salt, and encrypted data
    kdf_id = encrypted_data[magic_len:magic_len + 2]
    salt = encrypted_data[magic_len + 2:magic_len + 2 + 16]
    encrypted = encrypted_data[magic_len + 2 + 16:]

    # Derive key from password + stored salt
    force_pbkdf2 = (kdf_id != _KDF_ARGON2ID)
    key, _, _ = derive_key_from_passphrase(password, salt, force_pbkdf2=force_pbkdf2)

    # Decrypt
    try:
        plaintext = decrypt_bytes(key, encrypted)
    except Exception as exc:
        raise ValueError(f"Decryption failed (wrong password?): {exc}")

    # Parse JSON
    try:
        backup = json.loads(plaintext.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Decrypted data is not valid backup JSON: {exc}")

    return backup


def is_encrypted_backup(data: bytes) -> bool:
    """Check if data is an encrypted backup file (S125).

    Args:
        data: Raw file bytes.

    Returns:
        True if the data starts with the encrypted backup magic bytes.
    """
    return data[:len(_ENCRYPTED_MAGIC)] == _ENCRYPTED_MAGIC


# =====================================================================
# Module-level singleton
# =====================================================================

try:
    backup_manager = BackupManager()
    BACKUP_AVAILABLE = True
except Exception as _exc:
    logger.warning("BackupManager initialization failed: %s", _exc)
    backup_manager = None  # type: ignore[assignment]
    BACKUP_AVAILABLE = False
