"""Evaluator."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field


class StepResult(BaseModel):
    """
    Represents the result of a single validation step.
    """

    name: str
    passed: bool
    details: Optional[str] = Field(default=None)


class GeneratorResult(BaseModel):
    """
    Represents audit results for one generator folder by a specific auditor.
    """

    gen_name: str
    gen_id: str
    auditor: str
    passed: bool
    steps: List[StepResult] = Field(default_factory=list)


class AuditResult(BaseModel):
    """
    Aggregates results across all generators and auditors.
    """

    results: List[GeneratorResult] = Field(default_factory=list)
    passed_ids: List[str] = Field(default_factory=list)
    failed_ids: List[str] = Field(default_factory=list)


class BaseAuditor(ABC):
    """
    Abstract base class for generator auditors.
    All auditors must accept gen_path, gen_name, and optional validators,
    and implement the audit() method returning a GeneratorResult.
    Provides an async wrapper audit_async.
    """

    def __init__(
        self, gen_path: str, gen_name: str, validators: Optional[List[str]] = None
    ):
        self.gen_path = gen_path
        self.gen_name = gen_name
        self.validators = validators or []

    @abstractmethod
    def audit(self) -> GeneratorResult:
        """
        Perform all validation steps and return a GeneratorResult.
        """
        ...

    async def audit_async(self) -> GeneratorResult:
        """
        Asynchronous wrapper for audit, executing in default executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.audit)
