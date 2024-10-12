from abc import abstractmethod
from typing import List, Optional

from docarray import DocList

from marie.api.docs import MarieDoc
from marie.base_handler import BaseHandler
from marie.logging_core.logger import MarieLogger


class BaseDocumentBoundaryRegistration(BaseHandler):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.logger = MarieLogger(self.__class__.__name__).logger

    @abstractmethod
    def predict(
        self,
        documents: DocList[MarieDoc],
        registration_method: Optional[str],  # absolute, fit_to_page
        registration_point: tuple[int, int],
        margin_width: int,
        margin_height: int,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList:
        """
        Predict document boundaries. This method must be implemented by subclasses.

        :param documents:
        :param registration_method:
        :param registration_point:
        :param margin_width:
        :param margin_height:
        :param words:
        :param boxes:
        :param batch_size:
        """
        pass

    def run(
        self,
        documents: DocList,
        registration_method: str = "absolute",  # absolute, fit_to_page
        registration_point: tuple[int, int] = (10, 10),
        margin_width: int = 5,
        margin_height: int = 5,
        words: Optional[List[List[str]]] = None,
        boxes: Optional[List[List[List[int]]]] = None,
        batch_size: Optional[int] = None,
    ) -> DocList:
        """
        Run the document boundary registration on the given documents.

        :param documents: the documents to find the registration for
        :param registration_method: the method to use for registration (absolute, fit_to_page)
        :param registration_point: the point to use for registration (x, y)
        :param margin_width: the margin width to use for registration (in pixels, default: 5)
        :param margin_height: the margin height to use for registration (in pixels, default: 5)
        :param words: Optional list of words for each document, some models might require this
        :param boxes: Optional list of boxes for each document, some models might require this
        :param batch_size: Optional batch size to use for prediction
        :return: the registered documents as a DocList[MarieDoc]
        """

        if registration_method not in ["absolute", "fit_to_page"]:
            raise ValueError(f"Invalid registration method: {registration_method}")
        if registration_point is None:
            raise ValueError("Registration point must be provided")
        if not isinstance(registration_point, tuple) or len(registration_point) != 2:
            raise ValueError("Registration point must be a tuple of two integers")

        if documents:
            results = self.predict(
                documents=documents,
                registration_method=registration_method,
                registration_point=registration_point,
                margin_width=margin_width,
                margin_height=margin_height,
                words=words,
                boxes=boxes,
                batch_size=batch_size,
            )
        else:
            results = DocList()

        document_id = [document.id for document in documents]
        self.logger.info(f"Registered documents with IDs: {document_id}")
        return results
