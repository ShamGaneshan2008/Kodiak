import asyncio
import logging
from abc import ABC, abstractmethod

from .base import BaseEvent

logger = logging.getLogger(__name__)


class EventPublisherBackend(ABC):
    @abstractmethod
    async def publish(self, event: BaseEvent) -> bool:
        pass

    @abstractmethod
    async def batch_publish(self, events: list[BaseEvent]) -> int:
        pass


class EventPublisher:
    def __init__(self):
        self.backends: dict[str, EventPublisherBackend] = {}
        self.routes: dict[str, list[str]] = {}
        self.failed_events: list[BaseEvent] = []
        self.max_retries = 3

    def register_backend(self, name: str, backend: EventPublisherBackend) -> None:
        self.backends[name] = backend
        logger.info(f"Registered event backend: {name}")

    def add_route(self, event_type: str, backend_names: list[str]) -> None:
        self.routes[event_type] = backend_names

    def remove_route(self, event_type: str) -> None:
        if event_type in self.routes:
            del self.routes[event_type]

    async def publish(
        self,
        event: BaseEvent,
        backends: list[str] | None = None,
    ) -> dict[str, bool]:
        target_backends = backends or self.routes.get(event.event_type, [])
        results = {}

        for backend_name in target_backends:
            if backend_name not in self.backends:
                logger.warning(f"Backend {backend_name} not registered")
                results[backend_name] = False
                continue

            backend = self.backends[backend_name]
            try:
                success = await self._publish_with_retry(backend, event)
                results[backend_name] = success
                if not success:
                    self.failed_events.append(event)
            except Exception as e:
                logger.error(f"Error publishing to {backend_name}: {e}")
                results[backend_name] = False
                self.failed_events.append(event)

        return results

    async def _publish_with_retry(
        self,
        backend: EventPublisherBackend,
        event: BaseEvent,
        attempt: int = 0,
    ) -> bool:
        try:
            return await backend.publish(event)
        except Exception as e:
            if attempt < self.max_retries:
                await asyncio.sleep(2**attempt)
                return await self._publish_with_retry(backend, event, attempt + 1)
            logger.error(f"Failed to publish after {self.max_retries} retries: {e}")
            return False

    async def batch_publish(
        self,
        events: list[BaseEvent],
        backends: list[str] | None = None,
    ) -> dict[str, int]:
        results = {}
        grouped_events: dict[str, list[BaseEvent]] = {}

        for event in events:
            target_backends = backends or self.routes.get(event.event_type, [])
            for backend_name in target_backends:
                if backend_name not in grouped_events:
                    grouped_events[backend_name] = []
                grouped_events[backend_name].append(event)

        for backend_name, backend_events in grouped_events.items():
            if backend_name not in self.backends:
                logger.warning(f"Backend {backend_name} not registered")
                results[backend_name] = 0
                continue

            backend = self.backends[backend_name]
            try:
                count = await backend.batch_publish(backend_events)
                results[backend_name] = count
            except Exception as e:
                logger.error(f"Error batch publishing to {backend_name}: {e}")
                results[backend_name] = 0
                self.failed_events.extend(backend_events)

        return results

    def get_failed_events(self) -> list[BaseEvent]:
        return self.failed_events

    async def retry_failed_events(self) -> int:
        if not self.failed_events:
            return 0

        retry_events = self.failed_events.copy()
        self.failed_events.clear()
        success_count = 0

        for event in retry_events:
            results = await self.publish(event)
            if any(results.values()):
                success_count += 1

        return success_count


class LogEventBackend(EventPublisherBackend):
    async def publish(self, event: BaseEvent) -> bool:
        logger.info(f"Event: {event.event_type} - {event.to_dict()}")
        return True

    async def batch_publish(self, events: list[BaseEvent]) -> int:
        for event in events:
            await self.publish(event)
        return len(events)


class WebhookEventBackend(EventPublisherBackend):
    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    async def publish(self, event: BaseEvent) -> bool:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=event.to_dict(),
                    headers=self.headers,
                    timeout=10,
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Webhook publish failed: {e}")
            return False

    async def batch_publish(self, events: list[BaseEvent]) -> int:
        success_count = 0
        for event in events:
            if await self.publish(event):
                success_count += 1
        return success_count
