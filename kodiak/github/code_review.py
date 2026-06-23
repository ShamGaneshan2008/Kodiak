import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .client import GitHubClient
from kodiak.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ReviewComment:
    path: str
    line: int
    body: str
    suggestion: Optional[str] = None


@dataclass
class CodeReview:
    pr_number: int
    summary: str
    comments: List[ReviewComment]
    approval: bool
    severity: str = "info"


class CodeReviewHandler:
    def __init__(self, github_client: GitHubClient, llm_provider: LLMProvider):
        self.client = github_client
        self.llm = llm_provider

    async def review_pull_request(
            self,
            owner: str,
            repo: str,
            pr_number: int,
    ) -> CodeReview:
        pr = await self.client.get_pull_request(owner, repo, pr_number)
        files = await self._get_changed_files(owner, repo, pr_number)

        comments = []
        for file_info in files:
            file_comments = await self._analyze_file(owner, repo, file_info)
            comments.extend(file_comments)

        summary = await self._generate_review_summary(comments)
        approval = len([c for c in comments if c.body.startswith("ERROR")]) == 0

        return CodeReview(
            pr_number=pr_number,
            summary=summary,
            comments=comments,
            approval=approval,
        )

    async def _get_changed_files(
            self,
            owner: str,
            repo: str,
            pr_number: int,
    ) -> List[Dict[str, Any]]:
        pr = await self.client.get_pull_request(owner, repo, pr_number)
        files = []

        try:
            url = f"{self.client.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
            async with self.client._get_session() as session:
                async with session.get(url, headers=self.client.headers) as resp:
                    files = await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch changed files: {e}")

        return files

    async def _analyze_file(
            self,
            owner: str,
            repo: str,
            file_info: Dict[str, Any],
    ) -> List[ReviewComment]:
        comments = []
        filename = file_info.get("filename", "")

        try:
            content = await self.client.get_file_content(owner, repo, filename)
            decoded_content = content.get("content", "")

            issues = await self._check_code_quality(decoded_content, filename)

            for issue in issues:
                comments.append(ReviewComment(
                    path=filename,
                    line=issue.get("line", 1),
                    body=issue.get("message", ""),
                    suggestion=issue.get("suggestion"),
                ))
        except Exception as e:
            logger.error(f"Failed to analyze file {filename}: {e}")

        return comments

    async def _check_code_quality(
            self,
            content: str,
            filename: str,
    ) -> List[Dict[str, Any]]:
        issues = []

        if filename.endswith(".py"):
            if "import *" in content:
                issues.append({
                    "line": 1,
                    "message": "WARNING: Avoid wildcard imports",
                    "suggestion": "Import specific modules instead",
                })

            if "password" in content.lower() and "=" in content:
                issues.append({
                    "line": 1,
                    "message": "ERROR: Hardcoded credentials detected",
                    "suggestion": "Use environment variables or secrets manager",
                })

            if "TODO" in content or "FIXME" in content:
                issues.append({
                    "line": 1,
                    "message": "INFO: TODO/FIXME comments found",
                    "suggestion": "Consider creating an issue instead",
                })

        return issues

    async def _generate_review_summary(self, comments: List[ReviewComment]) -> str:
        errors = [c for c in comments if "ERROR" in c.body]
        warnings = [c for c in comments if "WARNING" in c.body]
        infos = [c for c in comments if "INFO" in c.body]

        summary = f"Code Review Summary:\n"
        summary += f"- Errors: {len(errors)}\n"
        summary += f"- Warnings: {len(warnings)}\n"
        summary += f"- Info: {len(infos)}\n"

        if not comments:
            summary += "\n✓ No issues found!"

        return summary

    async def post_review(
            self,
            owner: str,
            repo: str,
            review: CodeReview,
    ) -> bool:
        try:
            await self.client.create_review(
                owner,
                repo,
                review.pr_number,
                review.summary,
                event="APPROVE" if review.approval else "REQUEST_CHANGES",
            )

            for comment in review.comments:
                if comment.line > 0:
                    await self.client.create_review_comment(
                        owner,
                        repo,
                        review.pr_number,
                        "",
                        comment.path,
                        comment.line,
                        comment.body,
                    )

            logger.info(f"Review posted for PR #{review.pr_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to post review: {e}")
            return False