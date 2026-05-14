"""
Decay functions for memory strength calculation.

Models how memories fade over time, inspired by:
- Ebbinghaus forgetting curve
- Power law of forgetting
- ACT-R memory model
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryStrengthFactors:
    """Factors that influence memory strength."""
    base_importance: float  # 0.0 to 1.0
    age_hours: float  # Hours since last access
    access_count: int  # Number of times accessed
    emotional_intensity: float  # 0.0 to 1.0
    reinforcement_count: int  # Times the memory was reinforced


class DecayFunction(ABC):
    """Abstract base class for decay functions."""

    @abstractmethod
    def calculate_strength(self, factors: MemoryStrengthFactors) -> float:
        """
        Calculate current memory strength.

        Returns a value between 0.0 (forgotten) and 1.0 (vivid).
        """
        pass

    @abstractmethod
    def time_until_threshold(
        self,
        factors: MemoryStrengthFactors,
        threshold: float = 0.1,
    ) -> Optional[float]:
        """
        Estimate hours until strength drops below threshold.

        Returns None if memory won't decay below threshold.
        """
        pass


# Removed decay functions (preserved formulas for reference):
#
# ExponentialDecay (Ebbinghaus): strength = base * e^(-λt) + floor
#   where λ = decay_rate * (1 - emotional_intensity * emotion_factor), t in days
#
# PowerLawDecay (Anderson): strength = base * (1 + t)^(-d) + floor
#   where d = decay_exponent * (1 - emotional_intensity * emotion_factor), t in days


class AdaptiveDecay(DecayFunction):
    """
    Adaptive decay that adjusts based on memory characteristics.

    - Important memories decay slower
    - Frequently accessed memories decay slower
    - Emotional memories are more persistent
    - Recent memories have fast initial decay, then slow down
    """

    def __init__(
        self,
        base_halflife_hours: float = 168,  # 1 week default half-life
        importance_factor: float = 2.0,  # How much importance extends half-life
        access_factor: float = 0.5,  # How much access extends half-life
        emotion_factor: float = 1.5,  # How much emotion extends half-life
        floor: float = 0.02,
    ):
        self.base_halflife = base_halflife_hours
        self.importance_factor = importance_factor
        self.access_factor = access_factor
        self.emotion_factor = emotion_factor
        self.floor = floor

    def calculate_strength(self, factors: MemoryStrengthFactors) -> float:
        """Calculate strength using adaptive decay."""
        # Calculate effective half-life
        halflife = self.base_halflife

        # Importance extends half-life
        halflife *= (1 + factors.base_importance * self.importance_factor)

        # Access extends half-life (logarithmic)
        halflife *= (1 + math.log1p(factors.access_count) * self.access_factor)

        # Emotion extends half-life
        halflife *= (1 + factors.emotional_intensity * self.emotion_factor)

        # Reinforcement extends half-life
        halflife *= (1 + factors.reinforcement_count * 0.2)

        # Calculate decay using half-life
        # Strength = 0.5^(t/halflife)
        decay = math.pow(0.5, factors.age_hours / halflife)

        # Calculate final strength
        strength = factors.base_importance * decay + self.floor

        return min(1.0, max(0.0, strength))

    def time_until_threshold(
        self,
        factors: MemoryStrengthFactors,
        threshold: float = 0.1,
    ) -> Optional[float]:
        """Estimate time until strength drops below threshold."""
        current = self.calculate_strength(factors)

        if current <= threshold:
            return 0.0

        if self.floor >= threshold:
            return None

        # Calculate effective half-life (same as in calculate_strength)
        halflife = self.base_halflife
        halflife *= (1 + factors.base_importance * self.importance_factor)
        halflife *= (1 + math.log1p(factors.access_count) * self.access_factor)
        halflife *= (1 + factors.emotional_intensity * self.emotion_factor)
        halflife *= (1 + factors.reinforcement_count * 0.2)

        # Solve for t: base * 0.5^(t/halflife) + floor = threshold
        target = threshold - self.floor
        if target <= 0 or factors.base_importance <= 0:
            return None

        ratio = target / factors.base_importance
        if ratio >= 1:
            return 0.0

        # t = halflife * log2(base/target)
        hours = halflife * math.log2(1 / ratio)
        return max(0.0, hours - factors.age_hours)


# Default decay function
DEFAULT_DECAY = AdaptiveDecay()
