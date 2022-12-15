from typing import Any, Callable, Dict, MutableMapping, OrderedDict


class RouteHandler:

    _ROUTE_HANDLERS: MutableMapping[str, Callable] = OrderedDict()

    @staticmethod
    def register_route(handler: Callable) -> None:
        RouteHandler._ROUTE_HANDLERS[handler.__class__.__name__] = handler
