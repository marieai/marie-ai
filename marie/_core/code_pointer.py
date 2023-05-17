import importlib
import inspect
import os
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Callable, List, NamedTuple, Optional, Sequence, cast

import dagster._check as check
from dagster._core.errors import DagsterImportError, DagsterInvariantViolationError
from dagster._serdes import whitelist_for_serdes
from dagster._seven import get_import_error_message, import_module_from_path
from dagster._utils import alter_sys_path, hash_collection


class CodePointer(ABC):
    @abstractmethod
    def load_target(self) -> object:
        pass

    @abstractmethod
    def describe(self) -> str:
        pass

    @staticmethod
    def from_module(
        module_name: str, definition: str, working_directory: Optional[str]
    ) -> "ModuleCodePointer":
        check.str_param(module_name, "module_name")
        check.str_param(definition, "definition")
        check.opt_str_param(working_directory, "working_directory")
        return ModuleCodePointer(module_name, definition, working_directory)

    @staticmethod
    def from_python_package(
        module_name: str, attribute: str, working_directory: Optional[str]
    ) -> "PackageCodePointer":
        check.str_param(module_name, "module_name")
        check.str_param(attribute, "attribute")
        check.opt_str_param(working_directory, "working_directory")
        return PackageCodePointer(module_name, attribute, working_directory)

    @staticmethod
    def from_python_file(
        python_file: str, definition: str, working_directory: Optional[str]
    ) -> "FileCodePointer":
        check.str_param(python_file, "python_file")
        check.str_param(definition, "definition")
        check.opt_str_param(working_directory, "working_directory")
        return FileCodePointer(
            python_file=python_file,
            fn_name=definition,
            working_directory=working_directory,
        )


class FileCodePointer(
    NamedTuple(
        "_FileCodePointer",
        [("python_file", str), ("fn_name", str), ("working_directory", Optional[str])],
    ),
    CodePointer,
):
    def __new__(
        cls, python_file: str, fn_name: str, working_directory: Optional[str] = None
    ):
        return super(FileCodePointer, cls).__new__(
            cls,
            check.str_param(python_file, "python_file"),
            check.str_param(fn_name, "fn_name"),
            check.opt_str_param(working_directory, "working_directory"),
        )

    def load_target(self) -> object:
        raise NotImplementedError()

    def describe(self) -> str:
        if self.working_directory:
            return "{self.python_file}::{self.fn_name} -- [dir {self.working_directory}]".format(
                self=self
            )
        else:
            return "{self.python_file}::{self.fn_name}".format(self=self)


class ModuleCodePointer(
    NamedTuple(
        "_ModuleCodePointer",
        [("module", str), ("fn_name", str), ("working_directory", Optional[str])],
    ),
    CodePointer,
):
    def __new__(
        cls, module: str, fn_name: str, working_directory: Optional[str] = None
    ):
        return super(ModuleCodePointer, cls).__new__(
            cls,
            check.str_param(module, "module"),
            check.str_param(fn_name, "fn_name"),
            check.opt_str_param(working_directory, "working_directory"),
        )

    def load_target(self) -> object:
        raise NotImplementedError()

    def describe(self) -> str:
        return "from {self.module} import {self.fn_name}".format(self=self)


class PackageCodePointer(
    NamedTuple(
        "_PackageCodePointer",
        [("module", str), ("attribute", str), ("working_directory", Optional[str])],
    ),
    CodePointer,
):
    def __new__(
        cls, module: str, attribute: str, working_directory: Optional[str] = None
    ):
        return super(PackageCodePointer, cls).__new__(
            cls,
            check.str_param(module, "module"),
            check.str_param(attribute, "attribute"),
            check.opt_str_param(working_directory, "working_directory"),
        )

    def load_target(self) -> object:
        raise NotImplementedError()

    def describe(self) -> str:
        return "from {self.module} import {self.attribute}".format(self=self)
