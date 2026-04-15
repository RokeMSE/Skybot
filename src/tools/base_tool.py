from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class ToolError(Exception):
    """Base exception for all tool-related errors."""
    pass


class ToolValidationError(ToolError):
    """Raised when tool input validation fails."""
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass


class ToolConfig(BaseModel):
    """
    Common configuration for tools.
    """
    name: str
    description: str = ""
    timeout_seconds: int = 500
    max_retries: int = 0
    retry_delay_seconds: float = 1.0


class BaseTool(ABC):
    """
    Base class for all tools.

    Every tool should:
    - define INPUT_MODEL
    - implement _run()
    - optionally override post_process()
    """

    INPUT_MODEL = None  # Subclasses should set this to a Pydantic model

    def __init__(self, config: ToolConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    def validate_input(self, input_data: Dict[str, Any]) -> BaseModel:
        """
        Validate input data using INPUT_MODEL.
        """
        if self.INPUT_MODEL is None:
            raise ToolValidationError(
                f"{self.__class__.__name__} must define INPUT_MODEL"
            )
        try:
            return self.INPUT_MODEL(**input_data)
        except ValidationError as e:
            raise ToolValidationError(
                f"Invalid input for tool '{self.name}': {e}"
            ) from e

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution entrypoint for all tools.
        Handles validation, retry, logging, and error normalization.
        """
        validated_input = self.validate_input(input_data)

        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                logger.info(
                    "Executing tool '%s' attempt %d with input=%s",
                    self.name,
                    attempt + 1,
                    validated_input.model_dump() if hasattr(validated_input, "model_dump") else validated_input.dict(),
                )

                result = self._run(validated_input)

                duration = time.time() - start_time
                logger.info(
                    "Tool '%s' executed successfully in %.3fs",
                    self.name,
                    duration,
                )

                return {
                    "success": True,
                    "tool_name": self.name,
                    "data": result,
                    "error": None,
                    "metadata": {
                        "duration_seconds": duration,
                        "attempt": attempt + 1,
                    },
                }

            except Exception as e:
                last_error = e
                logger.exception("Tool '%s' failed on attempt %d", self.name, attempt + 1)

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds)

        raise ToolExecutionError(
            f"Tool '{self.name}' failed after {self.config.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    @abstractmethod
    def _run(self, validated_input: BaseModel) -> Any:
        """
        Actual tool logic.
        Must be implemented by subclasses.
        """
        raise NotImplementedError