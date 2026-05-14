"""Base interface for memory stores."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class MemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    def save(self, item: Any) -> str:
        """Save an item, return its ID."""
        pass

    @abstractmethod
    def get(self, item_id: str) -> Optional[Any]:
        """Retrieve an item by ID."""
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete an item, return success."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        **filters
    ) -> List[Any]:
        """Search for items matching query."""
        pass

    @abstractmethod
    def list_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List all items with pagination."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of items."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all items, return count deleted."""
        pass
