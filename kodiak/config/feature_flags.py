from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureFlags:
    enable_learning: bool = True
    enable_auto_pr: bool = False
    require_human_approval: bool = True
