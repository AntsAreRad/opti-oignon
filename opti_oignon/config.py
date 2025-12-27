#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CENTRALIZED CONFIGURATION - OPTI-OIGNON 1.0
==========================================

Loads and manages configuration from YAML files.

This module is the central configuration point for Opti-Oignon.
All other modules import their config from here.

Author: Léon
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import logging

# Logger configuration
logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULT PATHS
# =============================================================================

# Project root (opti_oignon/ folder)
PROJECT_ROOT = Path(__file__).parent

# Configuration folder
CONFIG_DIR = PROJECT_ROOT / "config"

# Data folder for history and user data
DATA_DIR = PROJECT_ROOT / "data"

# Configuration files
MODELS_CONFIG_FILE = CONFIG_DIR / "models.yaml"
PRESETS_CONFIG_FILE = CONFIG_DIR / "presets.yaml"

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    Safely load a YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary with file contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is malformed
    """
    if not filepath.exists():
        logger.warning(f"Config file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            logger.debug(f"Config loaded from {filepath}")
            return data
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {}


def save_yaml(filepath: Path, data: Dict[str, Any]) -> bool:
    """
    Save a dictionary to a YAML file.
    
    Args:
        filepath: Destination path
        data: Data to save
        
    Returns:
        True if success, False otherwise
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.debug(f"Config saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class OptiOignonConfig:
    """
    Centralized Opti-Oignon configuration.
    
    Singleton that loads and exposes all configuration.
    
    Usage:
        config = OptiOignonConfig()
        model = config.get_model("code", "primary")
        temp = config.get_temperature("code")
    """
    
    _instance: Optional['OptiOignonConfig'] = None
    
    def __new__(cls):
        """Singleton pattern - only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration (only once)."""
        if self._initialized:
            return
            
        self._models_config: Dict = {}
        self._presets_config: Dict = {}
        self._user_config: Dict = {}
        
        self.reload()
        self._initialized = True
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._models_config = load_yaml(MODELS_CONFIG_FILE)
        self._presets_config = load_yaml(PRESETS_CONFIG_FILE)
        
        # Load user config if it exists
        user_config_file = DATA_DIR / "user_config.yaml"
        if user_config_file.exists():
            self._user_config = load_yaml(user_config_file)
        
        logger.info("Configuration reloaded")
    
    # -------------------------------------------------------------------------
    # Model Access
    # -------------------------------------------------------------------------
    
    # Cache pour éviter le spam de warnings
    _warned_types: set = set()
    
    def get_model(self, model_type: str, priority: str = "primary") -> str:
        """
        Get a model by type and priority.
        
        Args:
            model_type: Model type (code_r, code_python, reasoning, general, quick, etc.)
            priority: Priority (primary, quality, fast)
            
        Returns:
            Ollama model name
        """
        # Utiliser la section 'routing' pour le mapping type → modèle
        routing = self._models_config.get("routing", {})
        type_models = routing.get(model_type, {})
        
        model = type_models.get(priority)
        if model:
            return model
        
        # Fallback to primary if priority not found
        model = type_models.get("primary")
        if model:
            logger.debug(f"Fallback to primary for {model_type}/{priority}")
            return model
        
        # Last resort: first available fallback (avec cache pour le warning)
        fallbacks = self._models_config.get("fallback_order", [])
        if fallbacks:
            if model_type not in self._warned_types:
                logger.warning(f"Type {model_type} not found in routing, using fallback {fallbacks[0]}")
                self._warned_types.add(model_type)
            return fallbacks[0]
        
        # Absolute default value
        return "qwen3-coder:30b"
    
    def get_fallback_models(self) -> List[str]:
        """Return the list of fallback models in order."""
        return self._models_config.get("fallback_order", [])
    
    def get_special_model(self, purpose: str) -> Optional[str]:
        """Get a special model (vision, math, embeddings)."""
        special = self._models_config.get("special", {})
        return special.get(purpose)
    
    def get_blacklisted_models(self) -> List[Dict[str, str]]:
        """Return the list of models to avoid with their reasons."""
        return self._models_config.get("blacklist", [])
    
    def is_blacklisted(self, model: str) -> bool:
        """Check if a model is blacklisted."""
        blacklist = self.get_blacklisted_models()
        return any(item.get("model") == model for item in blacklist)
    
    # -------------------------------------------------------------------------
    # Temperature Access
    # -------------------------------------------------------------------------
    
    def get_temperature(self, task_type: str) -> float:
        """
        Get the optimal temperature for a task type.
        
        Args:
            task_type: Task type (code, debug, reasoning, writing, general)
            
        Returns:
            Temperature (float between 0 and 1)
        """
        temps = self._models_config.get("temperatures", {})
        return temps.get(task_type, 0.5)  # 0.5 default
    
    # -------------------------------------------------------------------------
    # Timeout Access
    # -------------------------------------------------------------------------
    
    def get_timeout(self, timeout_type: str = "default") -> int:
        """
        Get a timeout in seconds.
        
        Args:
            timeout_type: Timeout type (default, fast, deep)
            
        Returns:
            Timeout in seconds
        """
        timeouts = self._models_config.get("timeouts", {})
        return timeouts.get(timeout_type, 300)
    
    # -------------------------------------------------------------------------
    # Preset Access
    # -------------------------------------------------------------------------
    
    def get_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a preset by ID.
        
        Args:
            preset_id: Preset identifier (e.g., "r_fast")
            
        Returns:
            Dictionary with preset config, or None if not found
        """
        presets = self._presets_config.get("presets", {})
        return presets.get(preset_id)
    
    def get_all_presets(self) -> Dict[str, Dict[str, Any]]:
        """Return all available presets."""
        return self._presets_config.get("presets", {})
    
    def get_preset_display_order(self) -> List[str]:
        """Return the display order of presets."""
        return self._presets_config.get("display_order", list(self.get_all_presets().keys()))
    
    def get_auto_select_preset(self, task: str) -> Optional[str]:
        """
        Get the preset to auto-select for a task.
        
        Args:
            task: Detected task type
            
        Returns:
            Preset ID, or None
        """
        auto_select = self._presets_config.get("auto_select", {})
        return auto_select.get(task)
    
    def get_shortcuts(self) -> Dict[str, str]:
        """Return keyboard shortcuts to presets."""
        return self._presets_config.get("shortcuts", {})
    
    # -------------------------------------------------------------------------
    # User Configuration
    # -------------------------------------------------------------------------
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self._user_config.get(key, default)
    
    def set_user_preference(self, key: str, value: Any) -> bool:
        """
        Set a user preference and save it.
        
        Args:
            key: Preference key
            value: Value to store
            
        Returns:
            True if success
        """
        self._user_config[key] = value
        user_config_file = DATA_DIR / "user_config.yaml"
        return save_yaml(user_config_file, self._user_config)
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def as_dict(self) -> Dict[str, Any]:
        """Return all configuration as a dictionary."""
        return {
            "models": self._models_config,
            "presets": self._presets_config,
            "user": self._user_config,
        }
    
    def __repr__(self) -> str:
        model_count = len(self._models_config.get("models", {}))
        preset_count = len(self.get_all_presets())
        return f"<OptiOignonConfig: {model_count} model types, {preset_count} presets>"


# =============================================================================
# GLOBAL INSTANCE FOR EASY IMPORT
# =============================================================================

# Accessible singleton
config = OptiOignonConfig()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_model(model_type: str, priority: str = "primary") -> str:
    """Shortcut to config.get_model()."""
    return config.get_model(model_type, priority)


def get_temperature(task_type: str) -> float:
    """Shortcut to config.get_temperature()."""
    return config.get_temperature(task_type)


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    """Shortcut to config.get_preset()."""
    return config.get_preset(preset_id)


# =============================================================================
# STARTUP INITIALIZATION
# =============================================================================

def ensure_config_dirs():
    """Create configuration folders if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Create folders when module loads
ensure_config_dirs()


# =============================================================================
# DEBUG CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=== Opti-Oignon 1.0 Configuration ===\n")
    
    cfg = OptiOignonConfig()
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config dir: {CONFIG_DIR}")
    print(f"Data dir: {DATA_DIR}\n")
    
    print("--- Models by type ---")
    for mtype in ["code", "reasoning", "general", "quick"]:
        primary = cfg.get_model(mtype, "primary")
        fast = cfg.get_model(mtype, "fast")
        print(f"  {mtype}: primary={primary}, fast={fast}")
    
    print("\n--- Temperatures ---")
    for ttype in ["code", "debug", "reasoning", "writing", "general"]:
        print(f"  {ttype}: {cfg.get_temperature(ttype)}")
    
    print("\n--- Presets ---")
    for pid in cfg.get_preset_display_order()[:5]:
        preset = cfg.get_preset(pid)
        if preset:
            print(f"  {preset.get('icon', '•')} {pid}: {preset.get('name', 'N/A')}")
    print(f"  ... ({len(cfg.get_all_presets())} presets total)")
    
    print(f"\n{cfg}")
