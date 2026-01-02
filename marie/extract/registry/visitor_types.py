from typing import Type, TypeVar, Union

from marie.extract.engine.processing_visitor import ProcessingVisitor

# Allow registering either an instance or a class that produces an instance
TVisitor = TypeVar("TVisitor", bound=Union[ProcessingVisitor, Type[ProcessingVisitor]])
