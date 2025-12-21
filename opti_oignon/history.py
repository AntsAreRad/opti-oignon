#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HISTORY - OPTI-OIGNON 1.0
=========================

Conversation history management and export.

Features:
- Save conversations (question, refined prompt, response)
- Read and reuse past conversations
- Export to Markdown
- Search history

Author: Léon
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging

from .config import DATA_DIR, save_yaml, load_yaml

logger = logging.getLogger(__name__)

# =============================================================================
# HISTORY ENTRY STRUCTURE
# =============================================================================

@dataclass
class HistoryEntry:
    """A history entry."""
    id: str
    timestamp: str
    question: str
    refined_question: str
    response: str
    task_type: str
    model: str
    temperature: float
    preset_used: Optional[str] = None
    document: Optional[str] = None
    duration_seconds: float = 0.0
    rating: Optional[int] = None  # User rating 1-5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HistoryEntry':
        """Create an entry from a dictionary."""
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            question=data.get("question", ""),
            refined_question=data.get("refined_question", ""),
            response=data.get("response", ""),
            task_type=data.get("task_type", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.5),
            preset_used=data.get("preset_used"),
            document=data.get("document"),
            duration_seconds=data.get("duration_seconds", 0.0),
            rating=data.get("rating"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )
    
    def to_markdown(self) -> str:
        """Export entry to Markdown."""
        md = []
        md.append(f"## {self.timestamp}")
        md.append("")
        md.append(f"**Task:** {self.task_type} | **Model:** {self.model} | **Temp:** {self.temperature}")
        if self.preset_used:
            md.append(f"**Preset:** {self.preset_used}")
        if self.duration_seconds:
            md.append(f"**Duration:** {self.duration_seconds:.1f}s")
        if self.rating:
            md.append(f"**Rating:** {'⭐' * self.rating}")
        md.append("")
        
        md.append("### Original Question")
        md.append("")
        md.append(self.question)
        md.append("")
        
        if self.refined_question and self.refined_question != self.question:
            md.append("### Refined Question")
            md.append("")
            md.append(self.refined_question)
            md.append("")
        
        if self.document:
            md.append("### Document Provided")
            md.append("")
            md.append("```")
            md.append(self.document[:1000])
            if len(self.document) > 1000:
                md.append("... (truncated)")
            md.append("```")
            md.append("")
        
        md.append("### Response")
        md.append("")
        md.append(self.response)
        md.append("")
        
        if self.tags:
            md.append(f"**Tags:** {', '.join(self.tags)}")
            md.append("")
        
        md.append("---")
        md.append("")
        
        return "\n".join(md)


# =============================================================================
# HISTORY MANAGER
# =============================================================================

class HistoryManager:
    """
    Conversation history manager.
    
    Stores conversations in daily JSON files for
    easy navigation and export.
    
    Usage:
        history = HistoryManager()
        entry_id = history.add(question="...", response="...", ...)
        entries = history.get_recent(10)
        history.export_markdown(entries, "export.md")
    """
    
    def __init__(self):
        """Initialize the manager."""
        self._history_dir = DATA_DIR / "history"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._history_dir / "index.json"
        self._index: Dict[str, str] = {}  # id -> filename
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the entry index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._index = {}
    
    def _save_index(self) -> None:
        """Save the index."""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _get_daily_file(self, date: datetime = None) -> Path:
        """Return the file for a given date."""
        date = date or datetime.now()
        filename = f"history_{date.strftime('%Y-%m-%d')}.json"
        return self._history_dir / filename
    
    def _generate_id(self, question: str) -> str:
        """Generate a unique ID for an entry."""
        timestamp = datetime.now().isoformat()
        content = f"{timestamp}:{question}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    # -------------------------------------------------------------------------
    # Adding entries
    # -------------------------------------------------------------------------
    
    def add(
        self,
        question: str,
        refined_question: str,
        response: str,
        task_type: str,
        model: str,
        temperature: float = 0.5,
        preset_used: Optional[str] = None,
        document: Optional[str] = None,
        duration_seconds: float = 0.0,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Add an entry to history.
        
        Args:
            question: Original question
            refined_question: Question after refinement
            response: Model response
            task_type: Task type
            model: Model used
            temperature: Temperature used
            preset_used: Preset used (optional)
            document: Document provided (optional)
            duration_seconds: Generation duration
            tags: Tags for search
            metadata: Additional metadata
            
        Returns:
            ID of the created entry
        """
        entry_id = self._generate_id(question)
        timestamp = datetime.now().isoformat()
        
        entry = HistoryEntry(
            id=entry_id,
            timestamp=timestamp,
            question=question,
            refined_question=refined_question,
            response=response,
            task_type=task_type,
            model=model,
            temperature=temperature,
            preset_used=preset_used,
            document=document,
            duration_seconds=duration_seconds,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        # Save to daily file
        daily_file = self._get_daily_file()
        entries = []
        
        if daily_file.exists():
            try:
                with open(daily_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            except:
                entries = []
        
        entries.append(entry.to_dict())
        
        with open(daily_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        
        # Update index
        self._index[entry_id] = daily_file.name
        self._save_index()
        
        logger.debug(f"Entry added: {entry_id}")
        return entry_id
    
    # -------------------------------------------------------------------------
    # Retrieving entries
    # -------------------------------------------------------------------------
    
    def get(self, entry_id: str) -> Optional[HistoryEntry]:
        """
        Get an entry by its ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            The entry or None if not found
        """
        filename = self._index.get(entry_id)
        if not filename:
            return None
        
        filepath = self._history_dir / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry_data in entries:
                if entry_data.get("id") == entry_id:
                    return HistoryEntry.from_dict(entry_data)
        except Exception as e:
            logger.error(f"Error reading entry {entry_id}: {e}")
        
        return None
    
    def get_recent(self, n: int = 10) -> List[HistoryEntry]:
        """
        Get the n most recent entries.
        
        Args:
            n: Number of entries to retrieve
            
        Returns:
            List of entries ordered from most recent to oldest
        """
        all_entries = []
        
        # List files by date (most recent first)
        files = sorted(self._history_dir.glob("history_*.json"), reverse=True)
        
        for filepath in files:
            if len(all_entries) >= n:
                break
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                
                # Add entries in reverse order (most recent first)
                for entry_data in reversed(entries):
                    all_entries.append(HistoryEntry.from_dict(entry_data))
                    if len(all_entries) >= n:
                        break
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        return all_entries
    
    def get_by_date(self, date: datetime) -> List[HistoryEntry]:
        """
        Get all entries for a given date.
        
        Args:
            date: Date to search
            
        Returns:
            List of entries for that date
        """
        filepath = self._get_daily_file(date)
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            return [HistoryEntry.from_dict(e) for e in entries]
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return []
    
    def get_by_task(self, task_type: str, limit: int = 50) -> List[HistoryEntry]:
        """
        Get entries for a task type.
        
        Args:
            task_type: Task type to filter
            limit: Maximum number of entries
            
        Returns:
            List of matching entries
        """
        results = []
        
        for filepath in sorted(self._history_dir.glob("history_*.json"), reverse=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                
                for entry_data in reversed(entries):
                    if entry_data.get("task_type") == task_type:
                        results.append(HistoryEntry.from_dict(entry_data))
                        if len(results) >= limit:
                            return results
            except:
                continue
        
        return results
    
    def search(self, query: str, limit: int = 20) -> List[HistoryEntry]:
        """
        Search in history.
        
        Args:
            query: Search term
            limit: Maximum number of results
            
        Returns:
            List of matching entries
        """
        query = query.lower()
        results = []
        
        for filepath in sorted(self._history_dir.glob("history_*.json"), reverse=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                
                for entry_data in reversed(entries):
                    # Search in question, response and tags
                    searchable = " ".join([
                        entry_data.get("question", ""),
                        entry_data.get("response", ""),
                        " ".join(entry_data.get("tags", [])),
                    ]).lower()
                    
                    if query in searchable:
                        results.append(HistoryEntry.from_dict(entry_data))
                        if len(results) >= limit:
                            return results
            except:
                continue
        
        return results
    
    # -------------------------------------------------------------------------
    # Modifying entries
    # -------------------------------------------------------------------------
    
    def rate(self, entry_id: str, rating: int) -> bool:
        """
        Rate an entry (1-5 stars).
        
        Args:
            entry_id: Entry ID
            rating: Rating from 1 to 5
            
        Returns:
            True if successful
        """
        if rating < 1 or rating > 5:
            return False
        
        filename = self._index.get(entry_id)
        if not filename:
            return False
        
        filepath = self._history_dir / filename
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry in entries:
                if entry.get("id") == entry_id:
                    entry["rating"] = rating
                    break
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"Rating error: {e}")
            return False
    
    def add_tag(self, entry_id: str, tag: str) -> bool:
        """Add a tag to an entry."""
        filename = self._index.get(entry_id)
        if not filename:
            return False
        
        filepath = self._history_dir / filename
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry in entries:
                if entry.get("id") == entry_id:
                    tags = entry.get("tags", [])
                    if tag not in tags:
                        tags.append(tag)
                        entry["tags"] = tags
                    break
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"Error adding tag: {e}")
            return False
    
    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        filename = self._index.get(entry_id)
        if not filename:
            return False
        
        filepath = self._history_dir / filename
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            entries = [e for e in entries if e.get("id") != entry_id]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            
            del self._index[entry_id]
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Deletion error: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    
    def export_markdown(
        self, 
        entries: List[HistoryEntry], 
        filepath: Path,
        title: str = "Opti-Oignon History Export"
    ) -> bool:
        """
        Export entries to Markdown.
        
        Args:
            entries: List of entries to export
            filepath: Destination path
            title: Document title
            
        Returns:
            True if successful
        """
        try:
            md = []
            md.append(f"# {title}")
            md.append("")
            md.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
            md.append("")
            md.append(f"**{len(entries)} conversations**")
            md.append("")
            md.append("---")
            md.append("")
            
            for entry in entries:
                md.append(entry.to_markdown())
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(md))
            
            logger.info(f"Markdown export: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    def export_json(self, entries: List[HistoryEntry], filepath: Path) -> bool:
        """Export entries to JSON."""
        try:
            data = [entry.to_dict() for entry in entries]
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON export: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about history."""
        total = 0
        by_task = {}
        by_model = {}
        by_day = {}
        rated_sum = 0
        rated_count = 0
        
        for filepath in self._history_dir.glob("history_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                
                day = filepath.stem.replace("history_", "")
                by_day[day] = len(entries)
                
                for entry in entries:
                    total += 1
                    
                    task = entry.get("task_type", "unknown")
                    by_task[task] = by_task.get(task, 0) + 1
                    
                    model = entry.get("model", "unknown")
                    by_model[model] = by_model.get(model, 0) + 1
                    
                    rating = entry.get("rating")
                    if rating:
                        rated_sum += rating
                        rated_count += 1
            except:
                continue
        
        return {
            "total_entries": total,
            "by_task": by_task,
            "by_model": by_model,
            "by_day": by_day,
            "average_rating": rated_sum / rated_count if rated_count else None,
            "rated_count": rated_count,
        }
    
    def clear_old(self, days: int = 30) -> int:
        """
        Delete entries older than n days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted entries
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        deleted = 0
        
        for filepath in self._history_dir.glob("history_*.json"):
            # Extract date from filename
            try:
                date_str = filepath.stem.replace("history_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date.timestamp() < cutoff:
                    # Count entries before deletion
                    with open(filepath, 'r', encoding='utf-8') as f:
                        entries = json.load(f)
                    deleted += len(entries)
                    
                    # Delete file
                    filepath.unlink()
                    logger.info(f"File deleted: {filepath.name}")
            except:
                continue
        
        # Clean index
        self._load_index()
        
        return deleted


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

history = HistoryManager()


def add_entry(**kwargs) -> str:
    """Convenience function to add an entry."""
    return history.add(**kwargs)


def get_recent(n: int = 10) -> List[HistoryEntry]:
    """Convenience function to get recent entries."""
    return history.get_recent(n)


def search_history(query: str) -> List[HistoryEntry]:
    """Convenience function to search history."""
    return history.search(query)


# =============================================================================
# CLI FOR TESTS
# =============================================================================

if __name__ == "__main__":
    print("=== History Test ===\n")
    
    manager = HistoryManager()
    
    # Add a test entry
    entry_id = manager.add(
        question="How to calculate the mean in R?",
        refined_question="How to calculate the mean of a numeric vector in R with NA handling?",
        response="```r\nmean(x, na.rm = TRUE)\n```",
        task_type="code_r",
        model="qwen3-coder:30b",
        temperature=0.3,
        duration_seconds=2.5,
        tags=["r", "statistics"],
    )
    print(f"Entry added: {entry_id}")
    
    # Get recent entries
    recent = manager.get_recent(5)
    print(f"\n{len(recent)} recent entries:")
    for entry in recent:
        print(f"  [{entry.timestamp[:10]}] {entry.question[:50]}...")
    
    # Stats
    stats = manager.get_stats()
    print(f"\nStatistics:")
    print(f"  Total: {stats['total_entries']} entries")
    print(f"  By task: {stats['by_task']}")
