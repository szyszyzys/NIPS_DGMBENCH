from typing import List, Tuple

import torch

from marketplace.seller.seller import BaseSeller


class DataSeller(BaseSeller):
    """
    Seller that provides raw data points to the marketplace.
    Inherits price-based selection and statistics from BaseSeller.
    """

    def __init__(self,
                 seller_id: str,
                 dataset: List[Tuple[torch.Tensor, int]],
                 price_strategy: str = 'uniform',
                 base_price: float = 1.0,
                 price_variation: float = 0.2):
        super().__init__(
            seller_id=seller_id,
            dataset=dataset,
            price_strategy=price_strategy,
            base_price=base_price,
            price_variation=price_variation
        )
        # Any data-seller-specific initialization can go here if needed.

    # (Optionally, override get_data if you want different logic.)
    # Otherwise, the base .get_data is fine, returning { "X": self.cur_data, "cost": self.cur_price }.
