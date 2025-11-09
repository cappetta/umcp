"""Core functionality for Ultimate MCP Server."""

import asyncio
from typing import Optional, TYPE_CHECKING

from ultimate_mcp_server.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ultimate_mcp_server.core.server import Gateway


logger = get_logger(__name__)

_gateway_instance: Optional["Gateway"] = None


def _get_gateway_class():
    """Lazy importer to avoid importing heavy FastAPI dependencies on module load."""

    from ultimate_mcp_server.core.server import Gateway  # Local import

    return Gateway


async def async_init_gateway():
    """Asynchronously initialize the global gateway instance."""

    global _gateway_instance
    if _gateway_instance is None:
        Gateway = _get_gateway_class()
        _gateway_instance = Gateway("provider-manager")
        await _gateway_instance._initialize_providers()
    return _gateway_instance


def get_provider_manager():
    """Get the provider manager from the Gateway instance."""

    global _gateway_instance

    if _gateway_instance is None:
        Gateway = _get_gateway_class()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(async_init_gateway())
                logger.warning("Gateway instance requested before async init completed.")
                return {}
            logger.info("Synchronously initializing gateway for get_provider_manager.")
            _gateway_instance = Gateway("provider-manager")
            loop.run_until_complete(_gateway_instance._initialize_providers())
        except RuntimeError:
            logger.info("Synchronously initializing gateway for get_provider_manager (new loop).")
            _gateway_instance = Gateway("provider-manager")
            asyncio.run(_gateway_instance._initialize_providers())

    return _gateway_instance.providers if _gateway_instance else {}


def get_gateway_instance() -> Optional["Gateway"]:
    """Synchronously get the initialized gateway instance."""

    global _gateway_instance
    if _gateway_instance is None:
        logger.warning("get_gateway_instance() called before instance was initialized.")
    return _gateway_instance


__all__ = ["async_init_gateway", "get_gateway_instance", "get_provider_manager"]
