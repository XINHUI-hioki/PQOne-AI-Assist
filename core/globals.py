"""
globals.py
---------------------------------
Centralised singleton container for runtime configuration, file cache,
and conversation memory.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory
lc_memory = ConversationBufferMemory(return_messages=True)
@dataclass
class AppConfig:
    api_endpoint: str = "http://127.0.0.1:11434/api/generate"
    default_language: str = "English"
    nav_mode: bool = False
    memory_enabled: bool = False
    beginner_mode: bool = False
    temperature: float = 0.8
    top_p: float = 0.9
    freq_penalty: float = 1.1
    max_length: int = 512
    dip_threshold: float = 180.0
    thd_threshold: float = 5.0


@dataclass
class FileCache:
    harmonics_df: Optional[Any] = None
    dips_df: Optional[Any] = None
    uploaded_paths: List[str] = field(default_factory=list)


@dataclass
class ConversationMemory:
    history: List[Dict[str, str]] = field(default_factory=list)

class GlobalState:

    def __init__(self) -> None:
        self.config: AppConfig = AppConfig()
        self.file_cache: FileCache = FileCache()
        self.memory: ConversationMemory = ConversationMemory()
        self.append_to_doc: bool = False


GLOBAL = GlobalState()
config = AppConfig()

def set_beginner_mode(flag: bool) -> None:
    GLOBAL.config.beginner_mode = bool(flag)
