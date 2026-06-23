import logging
import jwt
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InstallationManager:
    def __init__(self, app_id: str, private_key: str):
        self.app_id = app_id
        self.private_key = private_key
        self.installations: Dict[str, Dict[str, Any]] = {}

    def generate_jwt(self) -> str:
        now = datetime.utcnow()
        payload = {
            "iss": self.app_id,
            "iat": now,
            "exp": now + timedelta(minutes=10),
        }
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        return token

    async def register_installation(
            self,
            installation_id: str,
            access_token: str,
            repositories: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            self.installations[installation_id] = {
                "access_token": access_token,
                "repositories": repositories or [],
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            }
            logger.info(f"Registered installation {installation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register installation: {e}")
            return False

    async def get_installation_token(self, installation_id: str) -> Optional[str]:
        if installation_id not in self.installations:
            return None

        inst_data = self.installations[installation_id]
        expires_at = datetime.fromisoformat(inst_data["expires_at"])

        if datetime.utcnow() > expires_at:
            logger.warning(f"Installation token expired for {installation_id}")
            return None

        return inst_data["access_token"]

    async def revoke_installation(self, installation_id: str) -> bool:
        if installation_id in self.installations:
            del self.installations[installation_id]
            logger.info(f"Revoked installation {installation_id}")
            return True
        return False

    def get_installation_repositories(self, installation_id: str) -> list:
        if installation_id in self.installations:
            return self.installations[installation_id].get("repositories", [])
        return []

    def get_all_installations(self) -> Dict[str, Dict[str, Any]]:
        return self.installations.copy()

    async def check_access(self, installation_id: str, repository: str) -> bool:
        if installation_id not in self.installations:
            return False

        repositories = self.installations[installation_id].get("repositories", [])
        return any(r.get("name") == repository for r in repositories)