"""Public algorithms and their behavior/critic decompositions.

Every training stage uses this registry so a user selects the algorithm once,
rather than reproducing its component settings by hand.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    mode: str
    behavior_mode: str
    critic_mode: str


ALGORITHM_SPECS = MappingProxyType({
    "OMSD": AlgorithmSpec("OMSD", "SEQ", "sequential", "joint"),
    "CTDE": AlgorithmSpec("CTDE", "CTDE", "independent", "joint"),
    "IND": AlgorithmSpec("IND", "IND", "independent", "independent"),
})


def get_algorithm_spec(name):
    if not isinstance(name, str) or name.upper() not in ALGORITHM_SPECS:
        raise ValueError("algorithm must be OMSD, CTDE, or IND")
    return ALGORITHM_SPECS[name.upper()]
