import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class GitHubClient:
    def __init__(self, token: str, base_url: str = "https://api.github.com"):
        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Kodiak-AI",
        }

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}"
        return await self._request("GET", url)

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}"
        return await self._request("GET", url)

    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {"state": state, "per_page": per_page}
        return await self._request("GET", url, params=params)

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        return await self._request("POST", url, json=data)

    async def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        **kwargs,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}"
        return await self._request("PATCH", url, json=kwargs)

    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        return await self.update_issue(owner, repo, issue_number, state="closed")

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        return await self._request("GET", url)

    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": per_page}
        return await self._request("GET", url, params=params)

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        }
        return await self._request("POST", url, json=data)

    async def update_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        **kwargs,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        return await self._request("PATCH", url, json=kwargs)

    async def merge_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_message: str = "",
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/merge"
        data = {}
        if commit_message:
            data["commit_message"] = commit_message
        return await self._request("PUT", url, json=data)

    async def get_commits(
        self,
        owner: str,
        repo: str,
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page}
        return await self._request("GET", url, params=params)

    async def create_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        data = {"body": body, "event": event}
        return await self._request("POST", url, json=data)

    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_id: str,
        path: str,
        position: int,
        body: str,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        data = {
            "commit_id": commit_id,
            "path": path,
            "position": position,
            "body": body,
        }
        return await self._request("POST", url, json=data)

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> dict[str, Any]:
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        return await self._request("GET", url, params=params)

    async def list_repo_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = "main",
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        return await self._request("GET", url, params=params)

    async def _request(
        self,
        method: str,
        url: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=self.headers,
                    json=json,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in [200, 201, 204]:
                        if resp.status == 204:
                            return {}
                        return await resp.json()
                    else:
                        error_data = await resp.json()
                        logger.error(f"GitHub API error: {resp.status} - {error_data}")
                        raise Exception(f"GitHub API error: {resp.status}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
