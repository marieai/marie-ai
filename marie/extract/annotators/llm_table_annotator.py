import os
from typing import Any, List

from marie.executor.extract.util import setup_table_directories
from marie.extract.annotators.base import AnnotatorCapabilities
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.extract.annotators.util import (
    ascan_and_process_images,
    route_llm_engine,
    scan_and_process_images,
)
from marie.extract.structures import UnstructuredDocument


class LLMTableAnnotator(LLMAnnotator):
    """
    LLM Table Annotator
    """

    def __init__(
        self,
        working_dir: str,
        annotator_conf: dict[str, Any],
        layout_conf: dict[str, Any],
    ):
        """Initialize the table annotator.

        Args:
            working_dir: Current working directory for the given multi-page document.
            annotator_conf: Configuration for the annotator.
            layout_conf: Configuration for the layout.
        """
        super().__init__(working_dir, annotator_conf, layout_conf)
        self.silence_exceptions = False
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _log_debug_info(self, table_output_dir):
        """Log debug information."""
        self.logger.info("Debugging Information:")
        self.logger.info(f"Working Directory: {self.working_dir}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Frames Directory: {self.frames_dir}")
        self.logger.info(f"Prompt Text: {self.prompt_text}")
        self.logger.info(f"Model Name: {self.model_name}")
        self.logger.info(f"Multimodal: {self.multimodal}")
        self.logger.info(f"Expect Output: {self.expect_output}")
        self.logger.info(f"Table Annotated Directory: {table_output_dir}")

    def annotate(self, document: UnstructuredDocument, frames: List):
        """Annotate tables in the document.

        Args:
            document: Document to process
            frames: List of frames to process
        """
        self.logger.info(f"Annotating {self.__class__.__name__}...")
        (
            htables_output_dir,
            table_src_dir,
            table_annotated_dir,
            table_annotated_fragments_dir,
            table_output_dir,
        ) = setup_table_directories(self.working_dir, self.name)

        # moved to a separate executor
        # conf = OmegaConf.create({"grounding": {"table": []}})
        # parse_tables(document, self.working_dir, src_dir=table_src_dir, conf=conf)
        # highlight_tables(document, frames, htables_output_dir)
        # extract_tables(document, frames, metadata={}, output_dir=table_annotated_dir)
        #
        if False and os.listdir(table_annotated_fragments_dir):
            self.logger.info(
                f"Output directory '{table_annotated_fragments_dir}' contains results. Skipping annotation..."
            )
            return

        self._log_debug_info(table_output_dir)

        engine = route_llm_engine(self.model_name, self.multimodal)
        scan_and_process_images(
            table_annotated_fragments_dir,
            table_output_dir,
            self.prompt_text,
            document,
            engine=engine,
            is_multimodal=self.multimodal,
            expect_output=self.expect_output,
        )

    async def aannotate(self, document: UnstructuredDocument, frames: List) -> None:
        """Asynchronously annotate tables in the document.

        Args:
            document: Document to process
            frames: List of frames to process
        """
        self.logger.info(f"Annotating {self.__class__.__name__}...")
        (
            htables_output_dir,
            table_src_dir,
            table_annotated_dir,
            table_annotated_fragments_dir,
            table_output_dir,
        ) = setup_table_directories(self.working_dir, self.name)

        # moved to a separate executor
        # conf = OmegaConf.create({"grounding": {"table": []}})
        # parse_tables(document, self.working_dir, src_dir=table_src_dir, conf=conf)
        # highlight_tables(document, frames, htables_output_dir)
        # extract_tables(document, frames, metadata={}, output_dir=table_annotated_dir)
        #
        if False and os.listdir(table_annotated_fragments_dir):
            self.logger.info(
                f"Output directory '{table_annotated_fragments_dir}' contains results. Skipping annotation..."
            )
            return

        self._log_debug_info(table_output_dir)

        engine = route_llm_engine(self.model_name, self.multimodal)
        await ascan_and_process_images(
            table_annotated_fragments_dir,
            table_output_dir,
            self.prompt_text,
            document,
            engine=engine,
            is_multimodal=self.multimodal,
            expect_output=self.expect_output,
        )

    def parse_output(self, raw_output: str):
        """Parse the raw output from the LLM.

        Args:
            raw_output: Raw output string from the LLM
        """
        # TODO: Implement parsing logic
        return raw_output

    @property
    def capabilities(self) -> list[AnnotatorCapabilities]:
        """Get the capabilities of this annotator.

        Returns:
            List of annotator capabilities
        """
        return [AnnotatorCapabilities.EXTRACTOR]
