from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class KnowledgeBasePort(ABC):
    @abstractmethod
    def get_knowledge_base(self) -> Tuple[str, Dict[str, Any]]:
        """
        Loads and caches the knowledge base, reloading only if the file changes.
        Returns a tuple of (raw_json_string, parsed_dict).
        """

    @abstractmethod
    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Searches the knowledge base for chunks relevant to the query.
        """