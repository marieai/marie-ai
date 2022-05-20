from typing import MutableMapping, OrderedDict, Any


class RouteHandler:

    _ROUTE_HANDLERS: MutableMapping[str, Any] = OrderedDict()

    @staticmethod
    def register_route(handler: Any) -> None:
        RouteHandler._ROUTE_HANDLERS[handler.__class__.__name__] = handler
