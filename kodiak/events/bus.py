import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .base import BaseEvent, EventBus

logger = logging.getLogger(__name__)


class EventManager(EventBus):
    def __init__(self, max_queue_size: int = 1000):
        self.subscribers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.event_history: list[BaseEvent] = []
        self.max_history = 10000
        self.running = False

    async def publish(self, event: BaseEvent) -> None:
        try:
            await self.event_queue.put(event)
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: {event.event_type}")

    async def subscribe(
        self,
        event_type: str,
        callback: Callable,
        filter_fn: Callable | None = None,
    ) -> str:
        subscription_id = f"{event_type}_{id(callback)}"
        self.subscribers[event_type].append(
            {
                "callback": callback,
                "filter": filter_fn,
                "id": subscription_id,
            }
        )
        logger.info(f"Subscribed to {event_type} with id {subscription_id}")
        return subscription_id

    async def unsubscribe(self, event_type: str, subscription_id: str) -> None:
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                sub for sub in self.subscribers[event_type] if sub["id"] != subscription_id
            ]
            logger.info(f"Unsubscribed {subscription_id} from {event_type}")

    async def emit(self, event: BaseEvent) -> None:
        await self.publish(event)
        if event.event_type in self.subscribers:
            subscribers = self.subscribers[event.event_type]
            tasks = []
            for sub in subscribers:
                filter_fn = sub.get("filter")
                if filter_fn and not filter_fn(event):
                    continue
                callback = sub["callback"]
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(event))
                else:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in callback for {event.event_type}: {e}")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self) -> None:
        self.running = True
        asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        self.running = False

    async def _process_queue(self) -> None:
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.emit(event)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def clear(self) -> None:
        self.subscribers.clear()
        self.event_history.clear()
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def get_subscribers(self, event_type: str) -> list[str]:
        return [sub["id"] for sub in self.subscribers.get(event_type, [])]

    def get_history(self, event_type: str | None = None) -> list[BaseEvent]:
        if event_type:
            return [e for e in self.event_history if e.event_type == event_type]
        return self.event_history

    async def replay_history(self, event_type: str | None = None) -> None:
        events = self.get_history(event_type)
        for event in events:
            await self.emit(event)
