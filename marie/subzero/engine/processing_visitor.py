from abc import ABC, abstractmethod

from marie.subzero.models.definition import ExecutionContext
from marie.subzero.models.match import SubzeroResult


class ProcessingVisitor(ABC):
    """
    Document Processing Visitor
    """

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if this visitor is enabled

        :return: boolean indicating if the visitor is enabled
        """
        pass

    @abstractmethod
    def visit(self, context: "ExecutionContext", parent: "SubzeroResult") -> None:
        """
        Visit the given context

        :param context: GrapnelExecutionContext to visit
        :param parent: GrapnelResult parent node
        :raises: SubzeroException
        """
        pass

    @abstractmethod
    def pre_visit(self, context: "ExecutionContext", parent: "SubzeroResult") -> None:
        """
        This method gets invoked only once before visit

        :param context: ExecutionContext to visit
        :param parent: SubzeroResult parent node
        """
        pass

    @abstractmethod
    def post_visit(self, context: "ExecutionContext", parent: "SubzeroResult") -> None:
        """
        This method gets invoked only once after visit

        :param context: ExecutionContext to visit
        :param parent: SubzeroResult parent node
        """
        pass
