from typing import Mapping, Any, Optional, Sequence, NamedTuple, List

from marie import check


class NodeHandle(
    NamedTuple("_NodeHandle", [("name", str), ("parent", Optional["NodeHandle"])])
):
    """A structured object to identify nodes in the potentially recursive graph structure."""

    def __new__(cls, name: str, parent: Optional["NodeHandle"]):
        return super(NodeHandle, cls).__new__(
            cls,
            check.str_param(name, "name"),
            check.opt_inst_param(parent, "parent", NodeHandle),
        )

    def __str__(self):
        return self.to_string()

    @property
    def root(self):
        if self.parent:
            return self.parent.root
        else:
            return self

    @property
    def path(self) -> Sequence[str]:
        """Return a list representation of the handle.

        Inverse of NodeHandle.from_path.

        Returns:
            List[str]:
        """
        path: List[str] = []
        cur = self
        while cur:
            path.append(cur.name)
            cur = cur.parent
        path.reverse()
        return path

    def to_string(self) -> str:
        """Return a unique string representation of the handle.

        Inverse of NodeHandle.from_string.
        """
        return self.parent.to_string() + "." + self.name if self.parent else self.name

    def is_or_descends_from(self, handle: "NodeHandle") -> bool:
        """Check if the handle is or descends from another handle.

        Args:
            handle (NodeHandle): The handle to check against.

        Returns:
            bool:
        """
        check.inst_param(handle, "handle", NodeHandle)

        for idx in range(len(handle.path)):
            if idx >= len(self.path):
                return False
            if self.path[idx] != handle.path[idx]:
                return False
        return True

    def pop(self, ancestor: "NodeHandle") -> Optional["NodeHandle"]:
        """Return a copy of the handle with some of its ancestors pruned.

        Args:
            ancestor (NodeHandle): Handle to an ancestor of the current handle.

        Returns:
            NodeHandle:

        Example:
        .. code-block:: python

            handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
            ancestor = NodeHandle('bar', NodeHandle('foo', None))
            assert handle.pop(ancestor) == NodeHandle('baz', None)
        """
        check.inst_param(ancestor, "ancestor", NodeHandle)
        check.invariant(
            self.is_or_descends_from(ancestor),
            "Handle {handle} does not descend from {ancestor}".format(
                handle=self.to_string(), ancestor=ancestor.to_string()
            ),
        )

        return NodeHandle.from_path(self.path[len(ancestor.path) :])

    def with_ancestor(self, ancestor: Optional["NodeHandle"]) -> "NodeHandle":
        """Returns a copy of the handle with an ancestor grafted on.

        Args:
            ancestor (NodeHandle): Handle to the new ancestor.

        Returns:
            NodeHandle:

        Example:
        .. code-block:: pythonXXX

            handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
            ancestor = NodeHandle('abc' None)
            assert handle.with_ancestor(ancestor) == NodeHandle(
                'baz', NodeHandle('bar', NodeHandle('foo', NodeHandle('quux', None)))
            )
        """
        check.opt_inst_param(ancestor, "ancestor", NodeHandle)

        return NodeHandle.from_path([*(ancestor.path if ancestor else []), *self.path])

    @staticmethod
    def from_path(path: Sequence[str]) -> "NodeHandle":
        check.sequence_param(path, "path", of_type=str)

        cur: Optional["NodeHandle"] = None
        _path = list(path)
        while len(_path) > 0:
            cur = NodeHandle(name=_path.pop(0), parent=cur)

        if cur is None:
            check.failed(f"Invalid handle path {path}")

        return cur

    @staticmethod
    def from_string(handle_str: str) -> "NodeHandle":
        check.str_param(handle_str, "handle_str")

        path = handle_str.split(".")
        return NodeHandle.from_path(path)

    @classmethod
    def from_dict(cls, dict_repr: Mapping[str, Any]) -> "NodeHandle":
        """This method makes it possible to load a potentially nested NodeHandle after a
        roundtrip through json.loads(json.dumps(NodeHandle._asdict())).
        """
        check.dict_param(dict_repr, "dict_repr", key_type=str)
        check.invariant(
            "name" in dict_repr,
            "Dict representation of NodeHandle must have a 'name' key",
        )
        check.invariant(
            "parent" in dict_repr,
            "Dict representation of NodeHandle must have a 'parent' key",
        )

        if isinstance(dict_repr["parent"], (list, tuple)):
            parent = NodeHandle.from_dict(
                {
                    "name": dict_repr["parent"][0],
                    "parent": dict_repr["parent"][1],
                }
            )
        else:
            parent = dict_repr["parent"]

        return NodeHandle(name=dict_repr["name"], parent=parent)
