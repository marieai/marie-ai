import logging
import os
from typing import Callable, List, Optional, Union

from marie.extract.audit.base import BaseAuditor, GeneratorResult, StepResult

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class GeneratorAuditor(BaseAuditor):
    """
    Generic auditor that runs a set of validators.
    Validators and folder sets can be customized via config.
    """

    def __init__(
        self,
        gen_path: str,
        gen_name: str,
        folder_sets: Optional[List[str]] = None,
        extra_validators: Optional[
            List[Callable[[], Union[StepResult, List[StepResult]]]]
        ] = None,
    ):
        super().__init__(gen_path, gen_name)
        self.ao = os.path.join(self.gen_path, 'agent-output')
        # Configurable list of folders to check for 3-file sets
        self.folder_sets = folder_sets or ["claims", "tables", "key-values", "remarks"]
        # Core validators
        self._validator_methods: List[
            Callable[[], Union[StepResult, List[StepResult]]]
        ] = []
        self._validator_methods.append(self._check_claim_hybrid)
        self._validator_methods.append(self._make_flat_set_validator())
        self._validator_methods.append(self._check_fragments)
        self._validator_methods.append(self._check_table_extract)
        if extra_validators:
            self._validator_methods.extend(extra_validators)

    def _make_flat_set_validator(self) -> Callable[[], List[StepResult]]:
        def validator():
            results: List[StepResult] = []
            for folder in self.folder_sets:
                dir_path = os.path.join(self.ao, folder)
                if not os.path.isdir(dir_path):
                    results.append(
                        StepResult(name=folder, passed=False, details='folder missing')
                    )
                    continue
                try:
                    files = os.listdir(dir_path)
                    js = sum(1 for f in files if f.endswith('.json'))
                    pn = sum(1 for f in files if f.endswith('.png'))
                    pt = sum(1 for f in files if f.endswith('_prompt.txt'))
                except OSError as e:
                    logger.exception(f"IO error listing directory {dir_path}")
                    results.append(
                        StepResult(name=folder, passed=False, details=str(e))
                    )
                    continue
                ok = js == pn == pt
                detail = f"json={js}, png={pn}, prompt={pt}" if not ok else f"sets={js}"
                results.append(StepResult(name=folder, passed=ok, details=detail))
            return results

        return validator

    def audit(self) -> GeneratorResult:
        steps: List[StepResult] = []
        if not os.path.isdir(self.ao):
            gen_id = self._safe_extract_gen_id()
            details = 'agent-output folder missing'
            logger.error(f"{self.gen_name}: {details}")
            return GeneratorResult(
                gen_name=self.gen_name,
                gen_id=gen_id,
                auditor=self.__class__.__name__,
                passed=False,
                steps=[StepResult(name='agent-output', passed=False, details=details)],
            )
        for method in self._validator_methods:
            try:
                result = method()
                if isinstance(result, list):
                    steps.extend(result)
                else:
                    steps.append(result)
            except Exception as e:
                logger.exception(
                    f"Error running validator {method.__name__} for {self.gen_name}"
                )
                steps.append(
                    StepResult(name=method.__name__, passed=False, details=str(e))
                )
        gen_id = self._safe_extract_gen_id()
        passed = all(step.passed for step in steps)
        return GeneratorResult(
            gen_name=self.gen_name,
            gen_id=gen_id,
            auditor=self.__class__.__name__,
            passed=passed,
            steps=steps,
        )

    def _check_claim_hybrid(self) -> StepResult:
        path = os.path.join(self.ao, 'claim-hybrid', 'annotations.json')
        exists = os.path.isfile(path)
        return StepResult(
            name='claim-hybrid/annotations.json',
            passed=exists,
            details=None if exists else 'missing annotations.json',
        )

    def _check_fragments(self) -> StepResult:
        frag_dir = os.path.join(self.ao, 'table_annotated', 'fragments')
        if not os.path.isdir(frag_dir):
            return StepResult(
                name='table_annotated/fragments', passed=False, details='folder missing'
            )
        try:
            pngs = [f for f in os.listdir(frag_dir) if f.endswith('.png')]
        except OSError as e:
            logger.exception(f"IO error listing fragments {frag_dir}")
            return StepResult(
                name='table_annotated/fragments', passed=False, details=str(e)
            )
        missing = sum(
            1
            for img in pngs
            if not os.path.isfile(
                os.path.join(frag_dir, f"{os.path.splitext(img)[0]}_INJECTED_TEXT.txt")
            )
        )
        ok = missing == 0
        detail = f"missing {missing} of {len(pngs)}" if not ok else f"count={len(pngs)}"
        return StepResult(name='table_annotated/fragments', passed=ok, details=detail)

    def _check_table_extract(self) -> StepResult:
        te_dir = os.path.join(self.ao, 'table-extract')
        if not os.path.isdir(te_dir):
            return StepResult(
                name='table-extract', passed=False, details='folder missing'
            )
        try:
            mds = [f for f in os.listdir(te_dir) if f.endswith('.md')]
        except OSError as e:
            logger.exception(f"IO error listing table-extract {te_dir}")
            return StepResult(name='table-extract', passed=False, details=str(e))
        missing = sum(
            1
            for md in mds
            for suffix in ['.png', '.png_prompt.txt']
            if not os.path.isfile(
                os.path.join(te_dir, f"{os.path.splitext(md)[0]}{suffix}")
            )
        )
        ok = missing == 0
        detail = f"missing {missing} of {len(mds)}" if not ok else f"count={len(mds)}"
        return StepResult(name='table-extract', passed=ok, details=detail)

    def _safe_extract_gen_id(self) -> str:
        try:
            for fname in os.listdir(self.gen_path):
                if fname.startswith(self.gen_name) and fname.endswith('.json'):
                    return fname.split('.')[0]
        except OSError as e:
            logger.exception(f"IO error listing generator path {self.gen_path}")
