"""CHUWIBOT application package."""

from importlib.machinery import PathFinder
import importlib.abc
import sys


class _RobotControllerLoader(importlib.abc.Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        creator = getattr(self.original_loader, "create_module", None)
        return creator(spec) if creator else None

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        from app.person_runtime import install_person_tracking
        install_person_tracking(module.robot)


class _RobotControllerFinder(importlib.abc.MetaPathFinder):
    target = "app.robot_controller"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = _RobotControllerLoader(spec.loader)
        return spec


if not any(isinstance(finder, _RobotControllerFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _RobotControllerFinder())
