from abc import ABC, abstractmethod
from typing import Dict, Any




class DataMarketplace(ABC):
    @abstractmethod
    def register_seller(self, seller_id: str, seller: Any):
        """Register a seller in the marketplace."""
        pass

    @abstractmethod
    def get_market_status(self) -> Dict:
        """Get current market status (e.g., number of sellers, stats, etc.)."""
        pass

