import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Callable, List, Optional, Type

from marie.extract.audit.base import (
    AuditResult,
    BaseAuditor,
    GeneratorResult,
    StepResult,
)

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class AuditManager:
    """
    Manages auditing across all generators in a root directory using BaseAuditor subclasses.
    Supports callback hooks for on_generator_result and pluggable reporting strategies.
    Provides both sync run() and async run_async().
    """

    def __init__(
        self,
        root_dir: str,
        auditor_classes: Optional[List[Type[BaseAuditor]]] = None,
        on_generator_result: Optional[Callable[[GeneratorResult], None]] = None,
    ):
        self.root_dir = root_dir
        # Automatically discover auditors if not explicitly provided
        self.auditor_classes = auditor_classes or BaseAuditor.__subclasses__()
        self.on_generator_result = on_generator_result
        self.audit_result: AuditResult = AuditResult()

    def run(self) -> AuditResult:
        self.audit_result = AuditResult()

        if self.auditor_classes is None or len(self.auditor_classes) == 0:
            logger.error("No auditor classes provided.")
            return self.audit_result

        for gen_name in sorted(os.listdir(self.root_dir)):
            gen_path = os.path.join(self.root_dir, gen_name)
            result = None
            for cls in self.auditor_classes:
                try:
                    auditor = cls(gen_path, gen_name)
                    result = auditor.audit()
                except Exception as e:
                    logger.exception(
                        f"Unexpected error in auditor {cls.__name__} for {gen_name}"
                    )
                    result = GeneratorResult(
                        gen_name=gen_name,
                        gen_id='unknown',
                        auditor=cls.__name__,
                        passed=False,
                        steps=[StepResult(name='audit', passed=False, details=str(e))],
                    )
            self.audit_result.results.append(result)
            if self.on_generator_result:
                self.on_generator_result(result)
            (
                self.audit_result.passed_ids
                if result.passed
                else self.audit_result.failed_ids
            ).append(result.gen_id)

        return self.audit_result

    async def run_async(self) -> AuditResult:
        self.audit_result = AuditResult()
        tasks = []
        for gen_name in sorted(os.listdir(self.root_dir)):
            gen_path = os.path.join(self.root_dir, gen_name)
            for cls in self.auditor_classes:
                auditor = cls(gen_path, gen_name)
                tasks.append(auditor.audit_async())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.exception("Auditor task raised exception")
                continue
            self.audit_result.results.append(res)
            if self.on_generator_result:
                self.on_generator_result(res)
            (
                self.audit_result.passed_ids
                if res.passed
                else self.audit_result.failed_ids
            ).append(res.gen_id)
        return self.audit_result

    def print_summary(self):
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Audit Summary for {now}")
        print(f"Passed IDs: {', '.join(self.audit_result.passed_ids) or 'none'}")
        print(f"Failed IDs: {', '.join(self.audit_result.failed_ids) or 'none'}")


async def main_async(
    root_dir: str,
    on_generator_result: Optional[Callable[[GeneratorResult], None]] = None,
) -> int:
    manager = AuditManager(root_dir, on_generator_result=on_generator_result)
    await manager.run_async()
    manager.print_summary()
    return 0 if not manager.audit_result.failed_ids else 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Audit generator folders for required files and consistency.'
    )
    parser.add_argument(
        'root_dir',
        nargs='?',
        default='/tmp/generators',
        help='Root directory containing generator folders',
    )
    parser.add_argument(
        '--async',
        dest='use_async',
        action='store_true',
        help='Run audit asynchronously',
    )
    args = parser.parse_args()
    if not os.path.isdir(args.root_dir):
        print(f"Error: {args.root_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Example hook: log each result
    def hook(result: GeneratorResult):
        logger.info(
            f"Completed audit for {result.gen_name} (ID: {result.gen_id}) - {'PASSED' if result.passed else 'FAILED'}"
        )

    if args.use_async:
        exit_code = asyncio.run(main_async(args.root_dir, on_generator_result=hook))
    else:
        manager = AuditManager(args.root_dir, on_generator_result=hook)
        manager.run()
        manager.print_summary()
        exit_code = 0 if not manager.audit_result.failed_ids else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
