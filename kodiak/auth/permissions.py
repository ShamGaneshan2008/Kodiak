from enum import StrEnum


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


def has_permission(grants: set[str], required: Permission) -> bool:
    return Permission.ADMIN in grants or required in grants or required.value in grants
