# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for building pipelines in anomalib.

This module provides the abstract base class for creating pipelines that can execute
jobs in a configurable way. Pipelines handle setting up runners, parsing configs,
and orchestrating job execution.

Example:
    >>> from anomalib.pipelines.components.base import Pipeline
    >>> class MyPipeline(Pipeline):
    ...     def _setup_runners(self, args: dict) -> list[Runner]:
    ...         # Configure and return list of runners
    ...         pass
    ...     def run(self, args: Namespace | None = None):
    ...         # Execute pipeline logic
    ...         pass

The base pipeline interface defines key methods that subclasses must implement:

- :meth:`_setup_runners`: Configure the runners that will execute pipeline jobs
- :meth:`run`: Execute the core pipeline logic

Pipelines can be used to implement workflows like training, inference, or
benchmarking by composing jobs and runners in a modular way.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from jsonargparse import ArgumentParser, Namespace
from rich import traceback

from anomalib.utils.logging import redirect_logs

from .runner import Runner

if TYPE_CHECKING:
    from anomalib.pipelines.types import PREV_STAGE_RESULT
traceback.install()

log_file = "runs/pipeline.log"
logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """Base class for pipeline."""

    def _get_args(self, args: Namespace | None) -> dict:
        """Get pipeline arguments by parsing the config file.

        Args:
            args (Namespace | None): Arguments to run the pipeline.

        Returns:
            dict: Pipeline arguments.
        """
        if args is None:
            logger.warning("No arguments provided, parsing arguments from command line.")
            parser = self.get_parser()
            args = parser.parse_args()

        with Path(args.config).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    @abstractmethod
    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline."""

    def run(self, args: Namespace | None = None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        pipeline_args = self._get_args(args)
        runners = self._setup_runners(pipeline_args)
        redirect_logs(log_file)
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                job_args = pipeline_args.get(runner.generator.job_class.name)
                previous_results = runner.run(job_args or {}, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    f" Please check {log_file} for more details.",
                )

    @staticmethod
    def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
        """Create a new parser if none is provided."""
        if parser is None:
            parser = ArgumentParser()
            parser.add_argument("--config", type=str | Path, help="Configuration file path.", required=True)

        return parser
