#!/usr/bin/env python3
"""
Red Team Report Generation — Opti-Oignon S148
================================================

Generates human-readable and machine-readable reports from campaign scores.

Formats:
- JSON   — structured data for programmatic consumption
- Text   — plain-text summary for terminal / logs
- Markdown — tables and sections for documentation

Reports include timestamp, config snapshot, per-target heatmap data,
and per-category / per-strategy / per-target breakdowns.
"""

__all__ = [
    "generate_json_report",
    "generate_text_report",
    "generate_markdown_report",
    "save_report",
]

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default output directory
_DEFAULT_OUTPUT_DIR = "data/redteam_results"


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def generate_json_report(
    campaign_score: Any,
    config_snapshot: dict[str, Any] | None = None,
    campaign_run: Any = None,
) -> dict[str, Any]:
    """Generate a structured JSON report from campaign scores.

    Parameters
    ----------
    campaign_score : CampaignScore
        Aggregated campaign metrics.
    config_snapshot : dict or None
        Configuration used for the campaign.
    campaign_run : CampaignRun or None
        Raw campaign run data (for timing info).

    Returns
    -------
    dict
        Complete JSON-serializable report.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {
        "report_type": "redteam_campaign",
        "timestamp": timestamp,
        "version": _get_version(),
        "summary": {
            "total_attacks": campaign_score.total,
            "total_bypasses": campaign_score.total_bypasses,
            "total_flags": campaign_score.total_flags,
            "total_blocks": campaign_score.total_blocks,
            "overall_bypass_rate": round(campaign_score.overall_bypass_rate, 4),
            "overall_detection_rate": round(
                campaign_score.overall_detection_rate, 4
            ),
            "overall_block_rate": round(campaign_score.overall_block_rate, 4),
        },
        "by_category": {
            k: v.to_dict()
            for k, v in campaign_score.by_category.items()
        },
        "by_target": {
            k: v.to_dict()
            for k, v in campaign_score.by_target.items()
        },
        "by_strategy": {
            k: v.to_dict()
            for k, v in campaign_score.by_strategy.items()
        },
        "heatmap": campaign_score.heatmap_data(),
    }

    if config_snapshot:
        report["config"] = config_snapshot

    if campaign_run is not None:
        report["timing"] = {
            "start_time": getattr(campaign_run, "start_time", 0.0),
            "end_time": getattr(campaign_run, "end_time", 0.0),
            "duration_seconds": round(
                getattr(campaign_run, "duration_seconds", 0.0), 2
            ),
            "errors_count": len(getattr(campaign_run, "errors", [])),
        }

    return report


# ---------------------------------------------------------------------------
# Plain text report
# ---------------------------------------------------------------------------

def generate_text_report(
    campaign_score: Any,
    config_snapshot: dict[str, Any] | None = None,
    campaign_run: Any = None,
) -> str:
    """Generate a human-readable plain-text summary.

    Parameters
    ----------
    campaign_score : CampaignScore
        Aggregated campaign metrics.
    config_snapshot : dict or None
        Configuration snapshot.
    campaign_run : CampaignRun or None
        Raw run data for timing.

    Returns
    -------
    str
        Formatted text report.
    """
    lines: list[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("=" * 60)
    lines.append("OPTI-OIGNON RED TEAM AUDIT REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Version:   {_get_version()}")

    if campaign_run is not None:
        duration = getattr(campaign_run, "duration_seconds", 0.0)
        errors = len(getattr(campaign_run, "errors", []))
        lines.append(f"Duration:  {duration:.1f}s")
        lines.append(f"Errors:    {errors}")

    lines.append("")

    # --- Overall summary ---
    lines.append("-" * 40)
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total attacks tested:  {campaign_score.total}")
    lines.append(f"Bypasses:              {campaign_score.total_bypasses}")
    lines.append(f"Flags:                 {campaign_score.total_flags}")
    lines.append(f"Blocks:                {campaign_score.total_blocks}")
    lines.append(
        f"Bypass rate:           "
        f"{campaign_score.overall_bypass_rate:.1%}"
    )
    lines.append(
        f"Detection rate:        "
        f"{campaign_score.overall_detection_rate:.1%}"
    )
    lines.append(
        f"Block rate:            "
        f"{campaign_score.overall_block_rate:.1%}"
    )
    lines.append("")

    # --- Per category ---
    if campaign_score.by_category:
        lines.append("-" * 40)
        lines.append("BY CATEGORY")
        lines.append("-" * 40)
        for name, bd in sorted(campaign_score.by_category.items()):
            lines.append(
                f"  {name:25s}  "
                f"total={bd.total:3d}  "
                f"bypass={bd.bypass_rate:.1%}  "
                f"detect={bd.detection_rate:.1%}  "
                f"block={bd.block_rate:.1%}"
            )
        lines.append("")

    # --- Per target ---
    if campaign_score.by_target:
        lines.append("-" * 40)
        lines.append("BY TARGET")
        lines.append("-" * 40)
        for name, bd in sorted(campaign_score.by_target.items()):
            lines.append(
                f"  {name:25s}  "
                f"total={bd.total:3d}  "
                f"bypass={bd.bypass_rate:.1%}  "
                f"detect={bd.detection_rate:.1%}  "
                f"block={bd.block_rate:.1%}"
            )
        lines.append("")

    # --- Per strategy ---
    if campaign_score.by_strategy:
        lines.append("-" * 40)
        lines.append("BY STRATEGY")
        lines.append("-" * 40)
        for name, bd in sorted(campaign_score.by_strategy.items()):
            lines.append(
                f"  {name:25s}  "
                f"total={bd.total:3d}  "
                f"bypass={bd.bypass_rate:.1%}  "
                f"detect={bd.detection_rate:.1%}"
            )
        lines.append("")

    # --- Config ---
    if config_snapshot:
        lines.append("-" * 40)
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        for k, v in sorted(config_snapshot.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    campaign_score: Any,
    config_snapshot: dict[str, Any] | None = None,
    campaign_run: Any = None,
) -> str:
    """Generate a Markdown report with tables.

    Parameters
    ----------
    campaign_score : CampaignScore
        Aggregated campaign metrics.
    config_snapshot : dict or None
        Configuration snapshot.
    campaign_run : CampaignRun or None
        Raw run data for timing.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines: list[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# Opti-Oignon Red Team Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Version:** {_get_version()}")

    if campaign_run is not None:
        duration = getattr(campaign_run, "duration_seconds", 0.0)
        errors = len(getattr(campaign_run, "errors", []))
        lines.append(f"**Duration:** {duration:.1f}s")
        lines.append(f"**Errors:** {errors}")

    lines.append("")

    # --- Overall summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total attacks | {campaign_score.total} |")
    lines.append(f"| Bypasses | {campaign_score.total_bypasses} |")
    lines.append(f"| Flags | {campaign_score.total_flags} |")
    lines.append(f"| Blocks | {campaign_score.total_blocks} |")
    lines.append(
        f"| Bypass rate | {campaign_score.overall_bypass_rate:.1%} |"
    )
    lines.append(
        f"| Detection rate | {campaign_score.overall_detection_rate:.1%} |"
    )
    lines.append(
        f"| Block rate | {campaign_score.overall_block_rate:.1%} |"
    )
    lines.append("")

    # --- Per category table ---
    if campaign_score.by_category:
        lines.append("## By Category")
        lines.append("")
        lines.append(
            "| Category | Total | Bypasses | Flags | Blocks "
            "| Bypass Rate | Detection Rate |"
        )
        lines.append(
            "|----------|-------|----------|-------|--------"
            "|-------------|----------------|"
        )
        for name, bd in sorted(campaign_score.by_category.items()):
            lines.append(
                f"| {name} | {bd.total} | {bd.bypasses} | {bd.flags} "
                f"| {bd.blocks} | {bd.bypass_rate:.1%} "
                f"| {bd.detection_rate:.1%} |"
            )
        lines.append("")

    # --- Per target table ---
    if campaign_score.by_target:
        lines.append("## By Target")
        lines.append("")
        lines.append(
            "| Target | Total | Bypasses | Flags | Blocks "
            "| Bypass Rate | Block Rate |"
        )
        lines.append(
            "|--------|-------|----------|-------|--------"
            "|-------------|------------|"
        )
        for name, bd in sorted(campaign_score.by_target.items()):
            lines.append(
                f"| {name} | {bd.total} | {bd.bypasses} | {bd.flags} "
                f"| {bd.blocks} | {bd.bypass_rate:.1%} "
                f"| {bd.block_rate:.1%} |"
            )
        lines.append("")

    # --- Per strategy table ---
    if campaign_score.by_strategy:
        lines.append("## By Strategy")
        lines.append("")
        lines.append(
            "| Strategy | Total | Bypasses | Flags | Blocks "
            "| Bypass Rate | Detection Rate |"
        )
        lines.append(
            "|----------|-------|----------|-------|--------"
            "|-------------|----------------|"
        )
        for name, bd in sorted(campaign_score.by_strategy.items()):
            lines.append(
                f"| {name} | {bd.total} | {bd.bypasses} | {bd.flags} "
                f"| {bd.blocks} | {bd.bypass_rate:.1%} "
                f"| {bd.detection_rate:.1%} |"
            )
        lines.append("")

    # --- Heatmap table ---
    heatmap = campaign_score.heatmap_data()
    if heatmap:
        lines.append("## Strategy × Target Heatmap")
        lines.append("")
        lines.append(
            "| Strategy | Target | Total | Bypasses | Bypass Rate |"
        )
        lines.append(
            "|----------|--------|-------|----------|-------------|"
        )
        for row in heatmap:
            lines.append(
                f"| {row['strategy']} | {row['target']} "
                f"| {row['total']} | {row['bypasses']} "
                f"| {row['bypass_rate']:.1%} |"
            )
        lines.append("")

    # --- Config ---
    if config_snapshot:
        lines.append("## Configuration")
        lines.append("")
        lines.append("```yaml")
        for k, v in sorted(config_snapshot.items()):
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_report(
    campaign_score: Any,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    config_snapshot: dict[str, Any] | None = None,
    campaign_run: Any = None,
    formats: list[str] | None = None,
) -> dict[str, str]:
    """Save reports to disk in requested formats.

    Parameters
    ----------
    campaign_score : CampaignScore
        Aggregated campaign metrics.
    output_dir : str or Path
        Directory to write reports into.
    config_snapshot : dict or None
        Config snapshot to embed.
    campaign_run : CampaignRun or None
        Run data for timing info.
    formats : list of str or None
        Formats to generate. Defaults to ["json", "text", "markdown"].

    Returns
    -------
    dict[str, str]
        Mapping of format name → file path written.
    """
    if formats is None:
        formats = ["json", "text", "markdown"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    saved: dict[str, str] = {}

    if "json" in formats:
        report = generate_json_report(
            campaign_score, config_snapshot, campaign_run
        )
        path = out / f"report_{timestamp_slug}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        saved["json"] = str(path)
        logger.info("JSON report saved to %s", path)

    if "text" in formats:
        text = generate_text_report(
            campaign_score, config_snapshot, campaign_run
        )
        path = out / f"report_{timestamp_slug}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        saved["text"] = str(path)
        logger.info("Text report saved to %s", path)

    if "markdown" in formats:
        md = generate_markdown_report(
            campaign_score, config_snapshot, campaign_run
        )
        path = out / f"report_{timestamp_slug}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        saved["markdown"] = str(path)
        logger.info("Markdown report saved to %s", path)

    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_version() -> str:
    """Get current Opti-Oignon version."""
    try:
        from opti_oignon.__version__ import __version__
        return __version__
    except ImportError:
        return "unknown"
