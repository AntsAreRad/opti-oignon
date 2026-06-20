#!/usr/bin/env python3
"""
Red Team Runner — Opti-Oignon S148
====================================

Orchestrates the full red team pipeline:
  attack generation → strategy application → target evaluation → scoring.

Supports:
- Full campaign mode (all categories × strategies × targets)
- Focused single-shot mode (one category × one strategy × one target)
- Progress callbacks for UI integration
- Async-capable parallel target execution
"""

__all__ = [
    "RedTeamRunner",
    "CampaignRun",
    "RunProgress",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RunProgress:
    """Progress snapshot during a campaign run.

    Attributes
    ----------
    total_steps : int
        Total number of attack × strategy × target combinations.
    completed_steps : int
        Number completed so far.
    current_category : str
        Category currently being processed.
    current_strategy : str
        Strategy currently being applied.
    current_target : str
        Target currently being tested.
    errors : int
        Number of errors encountered.
    """

    total_steps: int = 0
    completed_steps: int = 0
    current_category: str = ""
    current_strategy: str = ""
    current_target: str = ""
    errors: int = 0

    @property
    def percent(self) -> float:
        """Completion percentage (0.0–100.0)."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100.0

    @property
    def is_complete(self) -> bool:
        return self.completed_steps >= self.total_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "current_category": self.current_category,
            "current_strategy": self.current_strategy,
            "current_target": self.current_target,
            "errors": self.errors,
            "percent": round(self.percent, 1),
            "is_complete": self.is_complete,
        }


# Type alias for progress callback
ProgressCallback = Callable[[RunProgress], None]


@dataclass
class CampaignRun:
    """Result container for a full campaign run.

    Attributes
    ----------
    results : list
        List of (attack, strategy_name, target_result) tuples.
    config_snapshot : dict
        Configuration used for this run.
    start_time : float
        Unix timestamp when the run started.
    end_time : float
        Unix timestamp when the run finished.
    errors : list[str]
        Error messages encountered during execution.
    """

    results: list[tuple[Any, str, Any]] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run."""
        if self.end_time <= 0 or self.start_time <= 0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def total_attacks(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attacks": self.total_attacks,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors_count": len(self.errors),
            "config_snapshot": self.config_snapshot,
        }


class RedTeamRunner:
    """Orchestrates a complete red team audit campaign.

    Loads configuration, instantiates the attack generator, applies
    strategies, and routes payloads through each target adapter.

    Parameters
    ----------
    config : RedTeamConfig or None
        Red team configuration. If None, loads from default YAML.
    progress_callback : callable or None
        Called with a ``RunProgress`` snapshot after each step.
    """

    def __init__(
        self,
        config: Any = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if config is None:
            from .config import load_redteam_config
            config = load_redteam_config()

        self._config = config
        self._progress_callback = progress_callback
        self._progress = RunProgress()

        # Lazy-init components
        self._generator: Any = None
        self._targets: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Any:
        """Current configuration."""
        return self._config

    @property
    def progress(self) -> RunProgress:
        """Current progress snapshot."""
        return self._progress

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _ensure_generator(self) -> Any:
        """Lazy-init the AttackGenerator."""
        if self._generator is None:
            from .generator import AttackGenerator
            self._generator = AttackGenerator(
                model=self._config.model,
                ollama_url=self._config.ollama_url,
                min_attack_length=self._config.min_attack_length,
                max_attack_length=self._config.max_attack_length,
                seed_fallback=self._config.seed_fallback,
                seed_file=self._config.seed_file,
            )
        return self._generator

    def _ensure_target(self, target_name: str) -> Any:
        """Lazy-init a target adapter by name."""
        if target_name not in self._targets:
            from .targets import create_target
            kwargs: dict[str, Any] = {}
            if target_name == "chat":
                kwargs["model"] = self._config.model
                kwargs["ollama_url"] = self._config.ollama_url
            self._targets[target_name] = create_target(target_name, **kwargs)
        return self._targets[target_name]

    # ------------------------------------------------------------------
    # Active categories / strategies / targets
    # ------------------------------------------------------------------

    def _active_categories(self) -> list[str]:
        """Return categories enabled by config + toggles."""
        cats = list(self._config.categories)
        toggles = self._config.category_toggles
        if toggles:
            cats = [c for c in cats if toggles.get(c, True)]
        return cats

    def _active_strategies(self) -> list[str]:
        """Return strategy names from config."""
        return list(self._config.strategies)

    def _active_targets(self) -> list[str]:
        """Return target names from config."""
        return list(self._config.targets)

    # ------------------------------------------------------------------
    # Progress management
    # ------------------------------------------------------------------

    def _notify_progress(self) -> None:
        """Send progress update via callback."""
        if self._progress_callback is not None:
            try:
                self._progress_callback(self._progress)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run_single(
        self,
        category: str,
        strategy: str,
        target: str,
        count: int = 1,
    ) -> list[tuple[Any, str, Any]]:
        """Run a focused test: one category × one strategy × one target.

        Parameters
        ----------
        category : str
            Attack category name.
        strategy : str
            Strategy name to apply.
        target : str
            Target adapter name.
        count : int
            Number of attacks to generate.

        Returns
        -------
        list of (GeneratedAttack, strategy_name, TargetResult) tuples.
        """
        from .strategies import apply_strategy

        generator = self._ensure_generator()
        target_adapter = self._ensure_target(target)

        attacks = generator.generate_for_category(
            category,
            count=count,
            batch_size=min(count, self._config.batch_size),
        )

        results: list[tuple[Any, str, Any]] = []
        for attack in attacks:
            try:
                transformed = apply_strategy(
                    strategy,
                    attack.payload,
                    ollama_url=self._config.ollama_url,
                    model=self._config.model,
                )
                result = target_adapter.run(transformed)
                results.append((attack, strategy, result))
            except Exception as exc:
                logger.warning(
                    "run_single error cat=%s strat=%s target=%s: %s",
                    category, strategy, target, exc,
                )

        return results

    def run_campaign(self) -> CampaignRun:
        """Execute a full campaign: all categories × strategies × targets.

        Returns
        -------
        CampaignRun
            Complete results with timing and config snapshot.
        """
        from .strategies import apply_strategy

        categories = self._active_categories()
        strategies = self._active_strategies()
        targets = self._active_targets()

        generator = self._ensure_generator()

        # Calculate total steps
        # For each category we generate N attacks, then apply each strategy
        # to each attack, then run through each target.
        # Total = sum(attacks_generated × strategies × targets) per category.
        # Estimate: attacks_per_category × strategies × targets × categories
        estimated_steps = (
            self._config.attacks_per_category
            * len(strategies)
            * len(targets)
            * len(categories)
        )

        self._progress = RunProgress(total_steps=estimated_steps)
        self._notify_progress()

        campaign = CampaignRun(
            start_time=time.time(),
            config_snapshot={
                "model": self._config.model,
                "categories": categories,
                "strategies": strategies,
                "targets": targets,
                "attacks_per_category": self._config.attacks_per_category,
                "bypass_threshold": self._config.bypass_threshold,
                "flag_threshold": self._config.flag_threshold,
            },
        )

        for category in categories:
            self._progress.current_category = category

            # Generate attacks for this category
            try:
                attacks = generator.generate_for_category(
                    category,
                    count=self._config.attacks_per_category,
                    batch_size=self._config.batch_size,
                )
            except Exception as exc:
                msg = f"Generation failed for {category}: {exc}"
                logger.warning(msg)
                campaign.errors.append(msg)
                # Skip steps for this category
                skip = (
                    self._config.attacks_per_category
                    * len(strategies)
                    * len(targets)
                )
                self._progress.completed_steps += skip
                self._progress.errors += 1
                self._notify_progress()
                continue

            for strategy_name in strategies:
                self._progress.current_strategy = strategy_name

                for attack in attacks:
                    # Apply strategy
                    try:
                        transformed = apply_strategy(
                            strategy_name,
                            attack.payload,
                            ollama_url=self._config.ollama_url,
                            model=self._config.model,
                        )
                    except Exception as exc:
                        msg = (
                            f"Strategy {strategy_name} failed on "
                            f"{category}/{attack.hash}: {exc}"
                        )
                        logger.debug(msg)
                        campaign.errors.append(msg)
                        self._progress.completed_steps += len(targets)
                        self._progress.errors += 1
                        self._notify_progress()
                        continue

                    for target_name in targets:
                        self._progress.current_target = target_name

                        try:
                            target_adapter = self._ensure_target(target_name)
                            result = target_adapter.run(transformed)
                            campaign.results.append(
                                (attack, strategy_name, result)
                            )
                        except Exception as exc:
                            msg = (
                                f"Target {target_name} error: {exc}"
                            )
                            logger.debug(msg)
                            campaign.errors.append(msg)
                            self._progress.errors += 1

                        self._progress.completed_steps += 1
                        self._notify_progress()

        campaign.end_time = time.time()

        # Recalculate total to match actual
        self._progress.total_steps = self._progress.completed_steps
        self._notify_progress()

        logger.info(
            "Campaign complete: %d results, %d errors in %.1fs",
            campaign.total_attacks,
            len(campaign.errors),
            campaign.duration_seconds,
        )

        return campaign
