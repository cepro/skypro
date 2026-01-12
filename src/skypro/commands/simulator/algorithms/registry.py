"""Strategy registry for discovering and loading control strategies."""

import logging
from importlib.metadata import entry_points
from typing import Dict, Optional, Type

from skypro.commands.simulator.algorithms.base import ControlStrategy

logger = logging.getLogger(__name__)

# Entry point group name for strategy plugins
STRATEGY_ENTRY_POINT_GROUP = "skypro.strategies"

# Cache of discovered strategies
_strategy_cache: Optional[Dict[str, Type[ControlStrategy]]] = None


def discover_strategies() -> Dict[str, Type[ControlStrategy]]:
    """
    Discover all available strategies from entry points.

    Strategies are discovered from the 'skypro.strategies' entry point group.
    External packages can register strategies by adding entry points in their
    pyproject.toml:

        [project.entry-points."skypro.strategies"]
        myStrategy = "mypackage.strategies:MyStrategy"

    Returns:
        Dict mapping strategy names to strategy classes
    """
    global _strategy_cache

    if _strategy_cache is not None:
        return _strategy_cache

    _strategy_cache = {}

    # Discover strategies from entry points
    try:
        eps = entry_points(group=STRATEGY_ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                strategy_class = ep.load()
                _strategy_cache[ep.name] = strategy_class
                logger.debug(f"Discovered strategy: {ep.name} from {ep.value}")
            except Exception as e:
                logger.warning(f"Failed to load strategy '{ep.name}': {e}")
    except Exception as e:
        logger.warning(f"Failed to discover strategies: {e}")

    return _strategy_cache


def get_strategy(name: str) -> Optional[Type[ControlStrategy]]:
    """
    Get a strategy class by name.

    Args:
        name: The strategy name (as registered in entry points)

    Returns:
        The strategy class, or None if not found
    """
    strategies = discover_strategies()
    return strategies.get(name)


def list_strategies() -> list[str]:
    """
    List all available strategy names.

    Returns:
        List of strategy names
    """
    return list(discover_strategies().keys())


def clear_cache():
    """Clear the strategy cache. Useful for testing."""
    global _strategy_cache
    _strategy_cache = None
