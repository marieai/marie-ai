import os

from marie.base_handler import BaseHandler
from marie.constants import __model_path__


class BaseDocumentClassifier(BaseHandler):
    def __init__(
        self,
        work_dir: str,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.models_dir = os.path.join(models_dir, "classifier")
        self.cuda = cuda
        self.work_dir = work_dir
        self.initialized = False
