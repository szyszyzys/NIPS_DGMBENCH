from abc import ABC
from typing import List, Dict, Any

import numpy as np
import pandas as pd


class SellerStats:
    """Statistics for a seller's dataset and market performance"""
    total_points: int
    points_selected: int = 0
    selection_rate: float = 0.0
    market_share: float = 0.0
    revenue: float = 0.0
    avg_price: float = 0.0

    # If doing federated learning, you might also track:
    # rounds_participated: int = 0
    # rounds_selected: int = 0
    # etc.


class BaseSeller(ABC):
    """Enhanced base seller class with statistics tracking"""

    def __init__(self,
                 seller_id: str,
                 dataset,
                 price_strategy: str = 'uniform',
                 base_price: float = 1.0,
                 price_variation: float = 0.2,
                 save_path="",
                 device='cpu'):
        self.seller_id = seller_id
        self.dataset = dataset  # Full dataset (whether used for data selling or gradient).
        self.price_strategy = price_strategy
        self.base_price = base_price
        self.price_variation = price_variation
        self.device = device
        # Initialize statistics
        self.stats = SellerStats()
        self.stats.total_points = len(dataset)  # <--- important initialization

        # Generate initial prices (if relevant for data sellers)
        self.prices = self._generate_prices()

        # Histories of events (data selections, federated rounds, etc.)
        self.selection_history: List[Dict] = []  # e.g. data-buyer selections
        self.federated_round_history: List[Dict] = []  # e.g. fed-learning rounds

        # "Current" data and price that might be offered to the market
        self.cur_data = self.dataset
        self.cur_price = self.prices
        self.save_path = save_path

        # Path(self.exp_save_path).mkdir(parents=True, exist_ok=True)

    @property
    def get_data(self) -> Dict[str, Any]:
        """
        Return the data or relevant info for the marketplace.
        Override in subclasses if needed.
        """
        return {
            "X": self.cur_data,
            "cost": self.cur_price,
        }

    def _generate_prices(self) -> np.ndarray:
        """Generate prices based on a specified strategy."""
        if self.price_strategy == 'uniform':
            return np.random.uniform(
                self.base_price * (1 - self.price_variation),
                self.base_price * (1 + self.price_variation),
                size=len(self.dataset)
            )
        elif self.price_strategy == 'gaussian':
            return np.abs(np.random.normal(
                self.base_price,
                self.price_variation * self.base_price,
                size=len(self.dataset)
            ))
        else:
            raise ValueError(f"Unknown price strategy: {self.price_strategy}")

    def record_selection(self, indices: List[int], buyer_id: str):
        """
        Record data points selected by a buyer (for the data marketplace).
        Subclasses can override if they need special behavior.
        """
        if not indices:
            return  # no selection

        selection_record = {
            'event_type': 'data_selection',
            'buyer_id': buyer_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_points': len(indices),
            'total_cost': float(np.sum(self.cur_price[indices])),
            'indices': indices
        }
        self.selection_history.append(selection_record)

        # Update statistics
        self.stats.points_selected += len(indices)
        if self.stats.total_points > 0:
            self.stats.selection_rate = (
                    self.stats.points_selected / self.stats.total_points
            )
        self.stats.revenue += selection_record['total_cost']
        if self.stats.points_selected > 0:
            self.stats.avg_price = self.stats.revenue / self.stats.points_selected

    def get_statistics(self) -> Dict:
        """Get current statistics as a dictionary."""
        return {
            'seller_id': self.seller_id,
            'dataset_size': self.stats.total_points,
            'points_selected': self.stats.points_selected,
            'selection_rate': self.stats.selection_rate,
            'revenue': self.stats.revenue,
            'avg_price': self.stats.avg_price,
            'market_share': self.stats.market_share,
        }

    # def save_statistics(self):
    #     """Save statistics and selection/federated round history to a JSON file."""
    #     stats_data = {
    #         'statistics': self.get_statistics(),
    #         'selection_history': self.selection_history,
    #         'federated_round_history': self.federated_round_history,
    #     }
    #     output_path = f"{self.exp_save_path}/result.json"
    #     with open(output_path, 'w') as f:
    #         json.dump(stats_data, f, indent=2)

    @property
    def exp_save_path(self):
        return f'{self.save_path}/{self.seller_id}'
