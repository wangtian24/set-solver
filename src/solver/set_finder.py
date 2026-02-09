"""
Set-finding algorithm.

A valid Set consists of 3 cards where, for each attribute,
the values are either ALL THE SAME or ALL DIFFERENT.
"""

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import List, Tuple


class Shape(IntEnum):
    DIAMOND = 0
    OVAL = 1
    SQUIGGLE = 2


class Color(IntEnum):
    RED = 0
    GREEN = 1
    PURPLE = 2


class Number(IntEnum):
    ONE = 0
    TWO = 1
    THREE = 2


class Fill(IntEnum):
    SOLID = 0
    STRIPED = 1
    EMPTY = 2


@dataclass
class Card:
    """A Set card with 4 attributes."""
    shape: Shape
    color: Color
    number: Number
    fill: Fill
    
    # Optional: position in image (for visualization)
    bbox: Tuple[float, float, float, float] = None  # x, y, w, h
    
    def __hash__(self):
        return hash((self.shape, self.color, self.number, self.fill))
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return (self.shape == other.shape and 
                self.color == other.color and
                self.number == other.number and 
                self.fill == other.fill)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return attributes as tuple of ints."""
        return (self.shape, self.color, self.number, self.fill)
    
    @classmethod
    def from_tuple(cls, attrs: Tuple[int, int, int, int], bbox=None) -> "Card":
        """Create card from tuple of attribute indices."""
        return cls(
            shape=Shape(attrs[0]),
            color=Color(attrs[1]),
            number=Number(attrs[2]),
            fill=Fill(attrs[3]),
            bbox=bbox
        )
    
    def __repr__(self):
        n = ["one", "two", "three"][self.number]
        return f"{n} {self.fill.name.lower()} {self.color.name.lower()} {self.shape.name.lower()}(s)"


def is_valid_set(card1: Card, card2: Card, card3: Card) -> bool:
    """
    Check if three cards form a valid Set.
    
    For each attribute, the three values must be either:
    - All the same (e.g., all red)
    - All different (e.g., red, green, purple)
    """
    for attr in ['shape', 'color', 'number', 'fill']:
        values = [getattr(card1, attr), getattr(card2, attr), getattr(card3, attr)]
        unique = len(set(values))
        # Valid: all same (1 unique) or all different (3 unique)
        # Invalid: exactly 2 unique
        if unique == 2:
            return False
    return True


def find_all_sets(cards: List[Card]) -> List[Tuple[Card, Card, Card]]:
    """
    Find all valid Sets among the given cards.
    
    Uses brute force: check all C(n,3) combinations.
    For 12 cards: C(12,3) = 220 combinations - very fast.
    For 21 cards (max in real game): C(21,3) = 1330 combinations - still fast.
    """
    valid_sets = []
    for combo in combinations(cards, 3):
        if is_valid_set(*combo):
            valid_sets.append(combo)
    return valid_sets


def find_first_set(cards: List[Card]) -> Tuple[Card, Card, Card] | None:
    """Find the first valid Set, or None if no Set exists."""
    for combo in combinations(cards, 3):
        if is_valid_set(*combo):
            return combo
    return None


# --- Utilities ---

def generate_all_cards() -> List[Card]:
    """Generate all 81 unique Set cards."""
    cards = []
    for s in Shape:
        for c in Color:
            for n in Number:
                for f in Fill:
                    cards.append(Card(shape=s, color=c, number=n, fill=f))
    return cards


def card_to_index(card: Card) -> int:
    """Convert card to unique index (0-80)."""
    return (card.shape * 27 + card.color * 9 + card.number * 3 + card.fill)


def index_to_card(idx: int) -> Card:
    """Convert index (0-80) to card."""
    fill = idx % 3
    idx //= 3
    number = idx % 3
    idx //= 3
    color = idx % 3
    idx //= 3
    shape = idx
    return Card(Shape(shape), Color(color), Number(number), Fill(fill))


# --- Demo ---

if __name__ == "__main__":
    # Example: find sets in a random deal
    import random
    
    all_cards = generate_all_cards()
    print(f"Total cards in deck: {len(all_cards)}")
    
    # Deal 12 cards
    deal = random.sample(all_cards, 12)
    print(f"\nDealt {len(deal)} cards:")
    for i, card in enumerate(deal):
        print(f"  {i+1}. {card}")
    
    # Find all sets
    sets = find_all_sets(deal)
    print(f"\nFound {len(sets)} valid Set(s):")
    for i, (c1, c2, c3) in enumerate(sets):
        print(f"\n  Set {i+1}:")
        print(f"    - {c1}")
        print(f"    - {c2}")
        print(f"    - {c3}")
