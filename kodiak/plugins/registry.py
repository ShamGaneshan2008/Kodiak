from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from kodiak.plugins.interface import Plugin, PluginMetadata

logger = structlog.get_logger(__name__)


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._lock = threading.Lock()

    def register(self, plugin: Plugin) -> None:
        name = plugin.metadata.name
        with self._lock:
            if name in self._plugins:
                logger.warning("plugin_already_registered", name=name)
                return
            self._plugins[name] = plugin
            logger.info("plugin_registered", name=name, version=plugin.metadata.version)

    def unregister(self, name: str) -> None:
        with self._lock:
            if name not in self._plugins:
                logger.warning("plugin_not_found_for_unregistration", name=name)
                return
            del self._plugins[name]
            logger.info("plugin_unregistered", name=name)

    def get(self, name: str) -> Plugin | None:
        with self._lock:
            return self._plugins.get(name)

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._plugins

    def list_plugins(self) -> list[PluginMetadata]:
        with self._lock:
            return [p.metadata for p in self._plugins.values()]