from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import structlog

from kodiak.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    from kodiak.plugins.interface import Plugin

logger = structlog.get_logger(__name__)


class PluginLoader:
    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    async def load_plugin(self, module_path: str, class_name: str) -> Plugin:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error("plugin_module_import_failed", path=module_path, error=str(e))
            raise

        plugin_class = getattr(module, class_name, None)
        if plugin_class is None:
            logger.error(
                "plugin_class_not_found",
                path=module_path,
                class_name=class_name,
            )
            raise AttributeError(f"Class {class_name} not found in {module_path}")

        instance: Plugin = plugin_class()
        await instance.initialize()

        self._registry.register(instance)
        logger.info(
            "plugin_loaded",
            name=instance.metadata.name,
            module=module_path,
        )
        return instance

    async def unload_plugin(self, name: str) -> bool:
        plugin = self._registry.get(name)
        if plugin is None:
            logger.warning("plugin_not_found_for_unload", name=name)
            return False

        await plugin.shutdown()
        self._registry.unregister(name)

        module_name = type(plugin).__module__
        if module_name in sys.modules:
            del sys.modules[module_name]
            logger.debug("plugin_module_removed", module=module_name)

        logger.info("plugin_unloaded", name=name)
        return True

    async def reload_plugin(self, name: str, module_path: str, class_name: str) -> Plugin:
        await self.unload_plugin(name)
        return await self.load_plugin(module_path, class_name)
