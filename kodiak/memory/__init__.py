# kodiak/memory/__init__.py
"""Kodiak Memory System package.

Provides Working Memory, Short-Term Memory, Long-Term Memory (Episodic,
Semantic, Procedural), Retrieval, Ranking, Context Building, Persistence,
and Consolidation.
"""

from .consolidation import ConsolidationResult, ConsolidationStatus, MemoryConsolidator
from .context import MemoryContextBuilder
from .episodic import Episode, EpisodeRepository, EpisodeSearchResult, EpisodicMemory
from .errors import (
    EpisodeNotFoundError,
    FactNotFoundError,
    MemoryNotFoundError,
    MemoryPersistenceError,
    MemoryServiceError,
    ProcedureNotFoundError,
    ShortTermMemoryError,
    WorkingMemoryNotFoundError,
)
from .long_term import LongTermMemory
from .models import Memory, MemoryType, SearchResult
from .persistence import (
    InMemoryEpisodeRepository,
    InMemoryProcedureRepository,
    InMemorySemanticRepository,
    InMemoryShortTermMemoryRepository,
    InMemoryWorkingMemoryRepository,
    JSONFileMemoryPersistence,
)
from .procedural import (
    ProceduralMemory,
    Procedure,
    ProcedureRepository,
    ProcedureSearchResult,
    ProcedureStep,
)
from .ranking import MemoryRanker
from .retrieval import MemoryRetriever
from .semantic import (
    FactNotFoundError,
    SemanticEntity,
    SemanticMemory,
    SemanticRepository,
    SemanticSearchResult,
)
from .short_term import ShortTermMemory, ShortTermMemoryItem, ShortTermMemoryRepository
from .experience import EngineeringExperience, ExperienceExtractor, ExperienceSanitizer
from .integration import MemoryIntegration
from .service import MemoryService
from .working import WorkingMemory, WorkingMemoryItem, WorkingMemoryRepository, WorkingMemoryStatus

__all__ = [
    # Models
    "Memory",
    "MemoryType",
    "SearchResult",
    "WorkingMemoryItem",
    "WorkingMemoryStatus",
    "ShortTermMemoryItem",
    "Episode",
    "EpisodeSearchResult",
    "SemanticEntity",
    "SemanticSearchResult",
    "Procedure",
    "ProcedureStep",
    "ProcedureSearchResult",
    "ConsolidationResult",
    "ConsolidationStatus",
    # Subsystems & Managers
    "WorkingMemory",
    "ShortTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "LongTermMemory",
    "MemoryRetriever",
    "MemoryRanker",
    "MemoryContextBuilder",
    "MemoryConsolidator",
    "MemoryService",
    "EngineeringExperience",
    "ExperienceExtractor",
    "ExperienceSanitizer",
    "MemoryIntegration",
    # Repositories & Persistence
    "InMemoryWorkingMemoryRepository",
    "InMemoryShortTermMemoryRepository",
    "InMemoryEpisodeRepository",
    "InMemorySemanticRepository",
    "InMemoryProcedureRepository",
    "JSONFileMemoryPersistence",
    "WorkingMemoryRepository",
    "ShortTermMemoryRepository",
    "EpisodeRepository",
    "SemanticRepository",
    "ProcedureRepository",
    # Exceptions
    "MemoryServiceError",
    "MemoryNotFoundError",
    "WorkingMemoryNotFoundError",
    "EpisodeNotFoundError",
    "FactNotFoundError",
    "ProcedureNotFoundError",
    "ShortTermMemoryError",
    "MemoryPersistenceError",
]
