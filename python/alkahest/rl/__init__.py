"""
alkahest.rl — reinforcement-learning environments backed by the Alkahest CAS.

The ``core`` subpackage is framework-agnostic (generators, verifiers, curriculum).
Domain-specific environments live under ``envs`` and may depend on optional packages
such as ``verifiers`` (Prime Intellect).
"""

from alkahest.rl.core import (
    BaseGenerator,
    BaseVerifier,
    CurriculumScheduler,
    Rubric,
    RubricFn,
)

__all__ = [
    "BaseGenerator",
    "BaseVerifier",
    "CurriculumScheduler",
    "Rubric",
    "RubricFn",
]
