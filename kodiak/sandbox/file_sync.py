import asyncio
import io
import shutil
import tarfile
from pathlib import Path

import structlog
from pydantic import BaseModel

from kodiak.sandbox.docker_backend import DockerBackend, SandboxContainer

logger = structlog.get_logger(__name__)


class FileSyncResult(BaseModel):
    files_copied: int = 0
    bytes_transferred: int = 0
    success: bool = False


class FileSyncManager:
    def __init__(self, backend: DockerBackend) -> None:
        self._backend = backend

    async def sync_to_sandbox(
        self, container: SandboxContainer, local_path: Path, remote_path: str
    ) -> FileSyncResult:
        if not await asyncio.to_thread(local_path.exists):
            logger.error("local_path_missing", path=str(local_path))
            return FileSyncResult(success=False)

        success = await self._backend.copy_files(container, local_path, remote_path)

        files_copied = 0
        bytes_transferred = 0
        if await asyncio.to_thread(local_path.is_file):
            files_copied = 1
            bytes_transferred = await asyncio.to_thread(lambda: local_path.stat().st_size)
        elif await asyncio.to_thread(local_path.is_dir):
            files_copied, bytes_transferred = await asyncio.to_thread(
                self._count_files,
                local_path,
            )

        logger.info(
            "synced_to_sandbox",
            container_id=container.container_id,
            files=files_copied,
            bytes=bytes_transferred,
        )
        return FileSyncResult(
            files_copied=files_copied,
            bytes_transferred=bytes_transferred,
            success=success,
        )

    async def sync_from_sandbox(
        self, container: SandboxContainer, remote_path: str, local_path: Path
    ) -> FileSyncResult:
        client = await self._backend.get_client()

        try:

            def _get_archive() -> bytes:
                c = client.containers.get(container.container_id)
                bits, _ = c.get_archive(remote_path)
                return b"".join(bits)

            archive_bytes = await asyncio.to_thread(_get_archive)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            files_copied = 0
            bytes_transferred = len(archive_bytes)

            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r") as tar:
                tar.extractall(path=local_path, filter="data")
                files_copied = len(tar.getmembers())

            logger.info(
                "synced_from_sandbox",
                container_id=container.container_id,
                files=files_copied,
                bytes=bytes_transferred,
            )
            return FileSyncResult(
                files_copied=files_copied,
                bytes_transferred=bytes_transferred,
                success=True,
            )
        except Exception as e:
            logger.error("sync_from_sandbox_failed", error=str(e))
            return FileSyncResult(success=False)

    async def sync_directory(
        self, container: SandboxContainer, local_dir: Path, remote_dir: str
    ) -> FileSyncResult:
        return await self.sync_to_sandbox(container, local_dir, remote_dir)

    async def delete_synced_files(self, local_path: Path) -> bool:
        try:
            await asyncio.to_thread(self._delete_path, local_path)
            logger.info("synced_files_deleted", path=str(local_path))
            return True
        except Exception as e:
            logger.error("delete_synced_files_failed", path=str(local_path), error=str(e))
            return False

    @staticmethod
    def _count_files(local_path: Path) -> tuple[int, int]:
        files_copied = 0
        bytes_transferred = 0
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                files_copied += 1
                bytes_transferred += file_path.stat().st_size
        return files_copied, bytes_transferred

    @staticmethod
    def _delete_path(local_path: Path) -> None:
        if local_path.is_file():
            local_path.unlink()
        elif local_path.is_dir():
            shutil.rmtree(local_path)
