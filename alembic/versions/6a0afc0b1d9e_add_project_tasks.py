"""add project-backed task fields

Revision ID: 6a0afc0b1d9e
Revises: 59f4667ad764
Create Date: 2026-08-23
"""

from alembic import op
import sqlalchemy as sa


revision = "6a0afc0b1d9e"
down_revision = "59f4667ad764"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("tasks", "repository_id", existing_type=sa.UUID(), nullable=True)
    op.add_column("tasks", sa.Column("project_id", sa.UUID(), nullable=True))
    op.add_column(
        "tasks",
        sa.Column("task_type", sa.String(length=255), nullable=False, server_default="implementation"),
    )
    op.add_column("tasks", sa.Column("github_issue_number", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_tasks_project_id_projects", "tasks", "projects", ["project_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index(op.f("ix_tasks_project_id"), "tasks", ["project_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_tasks_project_id"), table_name="tasks")
    op.drop_constraint("fk_tasks_project_id_projects", "tasks", type_="foreignkey")
    op.drop_column("tasks", "github_issue_number")
    op.drop_column("tasks", "task_type")
    op.drop_column("tasks", "project_id")
    op.alter_column("tasks", "repository_id", existing_type=sa.UUID(), nullable=False)
