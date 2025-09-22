from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from marie.logging_core.predefined import default_logger as logger


@dataclass
class PlannerConf:
    """Represents a single query planner configuration."""

    name: str
    py_module: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("Planner name cannot be empty")
        if not self.py_module:
            raise ValueError("Planner py_module cannot be empty")


@dataclass
class QueryPlannersConf:
    """Represents the query planners configuration."""

    planners: List[PlannerConf]
    watch_wheels: bool = False
    wheel_directories: List[str] = None

    def __post_init__(self):
        if not isinstance(self.planners, list):
            raise ValueError("planners must be a list")

        # Check for duplicate names
        names = [p.name for p in self.planners]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate planner names: {set(duplicates)}")

    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'QueryPlannersConf':
        """Create QueryPlannersConf from YAML content."""
        try:
            data = yaml.safe_load(yaml_content)

            if not isinstance(data, dict):
                raise ValueError("YAML content must be a dictionary")

            if 'query_planners' not in data:
                raise ValueError("Missing 'query_planners' in configuration")

            query_planners_data = data['query_planners']
            watch_wheels = query_planners_data.get('watch_wheels', False)
            wheel_directories = query_planners_data.get('wheel_directories', [])

            if 'planners' not in query_planners_data:
                raise ValueError("Missing 'planners' in query_planners")

            planners_list = query_planners_data['planners']
            if planners_list is None:
                logger.warning("Missing 'planners' in query_planners")
                planners_list = []

            if not isinstance(planners_list, list):
                raise ValueError("planners must be a list")

            planners = []
            for i, planner_data in enumerate(planners_list):
                if not isinstance(planner_data, dict):
                    raise ValueError(f"Planner at index {i} must be a dictionary")

                if 'name' not in planner_data:
                    raise ValueError(f"Planner at index {i} missing 'name' field")

                if 'py_module' not in planner_data:
                    raise ValueError(f"Planner at index {i} missing 'py_module' field")

                planners.append(
                    PlannerConf(
                        name=planner_data['name'], py_module=planner_data['py_module']
                    )
                )

            return cls(
                planners=planners,
                watch_wheels=watch_wheels,
                wheel_directories=wheel_directories,
            )

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryPlannersConf':
        """Create QueryPlannersConf from dictionary."""
        if 'planners' not in data:
            raise ValueError("Missing 'planners' in data")

        planners_list = data['planners']

        if planners_list is None:
            logger.warning("Missing 'planners' in query_planners")
            planners_list = []

        if not isinstance(planners_list, list):
            raise ValueError("planners must be a list")

        planners = []
        for planner_data in planners_list:
            planners.append(
                PlannerConf(
                    name=planner_data['name'], py_module=planner_data['py_module']
                )
            )

        watch_wheels = data.get('watch_wheels', False)
        wheel_directories = data.get('wheel_directories', [])

        return cls(
            planners=planners,
            watch_wheels=watch_wheels,
            wheel_directories=wheel_directories,
        )

    def get_planner_by_name(self, name: str) -> PlannerConf:
        """Get a planner by name."""
        for planner in self.planners:
            if planner.name == name:
                return planner
        raise ValueError(f"Planner '{name}' not found")

    def has_planner(self, name: str) -> bool:
        """Check if a planner with the given name exists."""
        try:
            self.get_planner_by_name(name)
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'watch_wheels': self.watch_wheels,
            'wheel_directories': self.wheel_directories or [],
            'planners': [
                {'name': p.name, 'py_module': p.py_module} for p in self.planners
            ],
        }


# Example usage:
if __name__ == "__main__":
    yaml_content = """
    query_planners:
      watch_wheels: True
      wheel_directories:
        - /mnt/data/marie-ai/config/wheels    
      planners:
        - name: tid_100985
          py_module: grapnel_g5.query.tid_100985.query
        - name: tid_121880
          py_module: grapnel_g5.query.tid_121880.query
    """

    try:
        query_planners_conf = QueryPlannersConf.from_yaml(yaml_content)

        print(f"Loaded {len(query_planners_conf.planners)} planners:")
        for planner in query_planners_conf.planners:
            print(f"  - {planner.name}: {planner.py_module}")

    except (ValueError, yaml.YAMLError) as e:
        print(f"Configuration error: {e}")
