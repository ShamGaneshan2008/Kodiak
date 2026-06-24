import asyncio
import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ContainerConfig(BaseModel):
    image: str = "python:3.12-slim"
    cpu_limit: float = 1.0
    memory_limit: str = "512m"
    working_directory: str = "/workspace"


class SandboxContainer(BaseModel):
    container_id: str
    status: str = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DockerBackend:
    def __init__(self, default_config: ContainerConfig | None = None) -> None:
        self._default_config = default_config or ContainerConfig()
        self._client: Any = None

    async def get_client(self) -> Any:
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
            except ImportError:
                raise RuntimeError("Docker SDK is not installed")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Docker daemon: {e}")
        return self._client

    async def create_container(
        self, config: ContainerConfig | None = None
    ) -> SandboxContainer:
        cfg = config or self._default_config
        client = await self.get_client()

        def _create() -> SandboxContainer:
            container = client.containers.create(
                image=cfg.image,
                cpu_quota=int(cfg.cpu_limit * 100000),
                mem_limit=cfg.memory_limit,
                working_dir=cfg.working_directory,
                tty=True,
                detach=True,
            )
            return SandboxContainer(container_id=container.id, status="created")

        result = await asyncio.to_thread(_create)
        logger.info("container_created", container_id=result.container_id)
        return result

    async def start_container(self, container: SandboxContainer) -> SandboxContainer:
        client = await self.get_client()

        def _start() -> None:
            c = client.containers.get(container.container_id)
            c.start()

        await asyncio.to_thread(_start)
        logger.info("container_started", container_id=container.container_id)
        return container.model_copy(update={"status": "running"})

    async def stop_container(self, container: SandboxContainer) -> SandboxContainer:
        client = await self.get_client()

        def _stop() -> None:
            c = client.containers.get(container.container_id)
            c.stop(timeout=10)

        await asyncio.to_thread(_stop)
        logger.info("container_stopped", container_id=container.container_id)
        return container.model_copy(update={"status": "stopped"})

    async def remove_container(self, container: SandboxContainer) -> bool:
        client = await self.get_client()

        def _remove() -> None:
            c = client.containers.get(container.container_id)
            c.remove(force=True)

        try:
            await asyncio.to_thread(_remove)
            logger.info("container_removed", container_id=container.container_id)
            return True
        except Exception:
            logger.warning("container_remove_failed", container_id=container.container_id)
            return False

    async def execute_command(
        self, container: SandboxContainer, command: str
    ) -> tuple[int, str, str]:
        client = await self.get_client()

        def _exec() -> tuple[int, str, str]:
            c = client.containers.get(container.container_id)
            exec_result = c.exec_run(cmd=command, demux=True, shell=True)
            stdout = (
                exec_result.output[0].decode("utf-8")
                if exec_result.output and exec_result.output[0]
                else ""
            )
            stderr = (
                exec_result.output[1].decode("utf-8")
                if exec_result.output and exec_result.output[1]
                else ""
            )
            return exec_result.exit_code, stdout, stderr

        return await asyncio.to_thread(_exec)

    async def copy_files(
        self, container: SandboxContainer, src: Path, dst: str
    ) -> bool:
        client = await self.get_client()

        def _copy() -> bool:
            c = client.containers.get(container.container_id)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                if src.is_file():
                    tar.add(src, arcname=src.name)
                elif src.is_dir():
                    for file_path in src.rglob("*"):
                        if file_path.is_file():
                            tar.add(
                                file_path,
                                arcname=file_path.relative_to(src),
                            )
            buf.seek(0)
            c.put_archive(dst, buf.read())
            return True