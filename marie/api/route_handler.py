from typing import MutableMapping, OrderedDict, Any, Callable, Dict


class RouteHandler:

    _ROUTE_HANDLERS: MutableMapping[str, Callable] = OrderedDict[]

    @staticmethod
    def register_route(handler: Callable) -> None:
        RouteHandler._ROUTE_HANDLERS[handler.__class__.__name__] = handler
