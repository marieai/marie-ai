# Component Registry

This guide explains how to use, extend, and operate the ComponentRegistry responsible for discovery and orchestration of extraction components. 
It covers responsibilities, registration/lookup APIs, initialization flows, wheel-based dynamic loading, thread-safety, configuration, and best practices.

## Purpose

ComponentRegistry is a thin facade that composes the following component types:
- Parsers
- Validators
- Template builders
- Region processors

It provides:
- A unified registration API (decorators) for each component type
- Lazy initialization and auto-loading of core components
- Support for dynamically loading external components (module scanning and Python wheels)

## What it manages

- Parsers: Callable interfaces used to transform inputs into structured outputs.
- Validators: Instances of BaseValidator that can support different ValidationStage values.
- Template builders: Functions that build structured templates used by the pipeline.
- Region processors: Function or class-based processors for structured regions.

Region processors can be registered as:
- A function 
- A class implementing the RegionProcessorProto with a process(...) method
- A class type that implements the protocol (will be coerced to a callable)

The coerces ensure uniform call semantics regardless of function/class form.

## Thread-safety

- All mutation and most reads are guarded by an internal RLock, ensuring safe concurrent access in multi-threaded environments.

## Initialization modes

There are three key initialization flows:

1) Core components (auto-loaded by default)
- On first component access or when initialize_core_components() is called, core modules are imported for their side effects (they register themselves using the decorators).
- Controlled by auto_load_core flag (true by default).

2) External components via module import
- initialize_external_components(modules, strict=True) imports submodules beneath provided roots.
- strict=True raises on any failed import; strict=False logs failures and continues.

3) Configuration-driven initialization
- initialize_from_config(config) supports:
  - load_core_components: bool (default: true)
  - external_component_modules: List[str] (module roots to scan)
  - wheel_directories: List[str] (directories to scan/install wheels from)
  - watch_wheels: bool (default: true) to watch the directories for new wheels

Returns a summary dict with loaded/failed modules and totals per component type. May include wheel_results if any were processed.

## Wheel-based dynamic loading

- The registry integrates with a wheel manager and a directory watcher to:
  - Install existing wheels on startup for configured directories
  - Optionally watch directories for new wheels and load them dynamically
- get_registry_info() exposes:
  - installed_wheels metadata (package name, modules_count, install_time)
  - watched_directories

Note: cleanup() stops watchers and cleans the wheel manager.

## Registration API

Use decorators exposed at module level for convenience:
- register_parser(name)
- register_validator(name)
- register_template_builder(name)
- register_region_processor(name)

Or use methods on a ComponentRegistry instance.

Behavior:
- Parsers: duplicate names log a warning and overwrite
- Validators: duplicate names raise an error (BadConfigSource)
- Template builders: duplicate names raise an error (BadConfigSource)
- Region processors: duplicate names log a warning and overwrite

You can also register a validator instance directly:
- register_validator_instance(validator: BaseValidator)

### Examples

Register a parser:
```python

from marie.extract.registry.base import register_parser

@register_parser("my_parser")
def my_parser(context, input_data):
    # parsing logic
    return {"ok": True}
```


Register a validator (factory/class/instance supported via coercion):
```python

from marie.extract.registry.base import register_validator
from marie.extract.validator.base import BaseValidator, ValidationStage

class MyValidator(BaseValidator):
    name = "my_validator"
    def supports_stage(self, stage: ValidationStage) -> bool:
        return stage == ValidationStage.POST
    def validate(self, data):
        # validation logic
        return []

@register_validator("my_validator")
class MyValidatorFactory:
    # returned/constructed object must be a BaseValidator-compatible instance
    def __call__(self):
        return MyValidator()
```


Register a template builder:
```python

from marie.extract.registry.base import register_template_builder

@register_template_builder("invoice_template")
def build_invoice_template(config):
    return {"template": "invoice", "config": config}
```


Register a region processor (function form):
```python

from marie.extract.registry.base import register_region_processor

@register_region_processor("remarks_processor")
def remarks_processor(context, parent_section, region_parser_config, regions_config):
    # produce structured region outputs
    return [{"type": "remarks", "items": []}]
```


Register a region processor (class form, Protocol):
```python

from marie.extract.registry.base import register_region_processor

@register_region_processor("totals_processor")
class TotalsProcessor:
    def process(self, context, parent_section, region_parser_config, regions_config):
        # class-based processing
        return [{"type": "totals", "items": []}]
```

## Configuration-driven initialization

Call initialize_from_config(config: Dict[str, Any]).

Supported keys:
- load_core_components: bool (default true)
- external_component_modules: List[str] (module roots scanned for components that register themselves)
- wheel_directories: List[str] (directories to scan for .whl files and install)
- watch_wheels: bool (default true)

Returns a dict summary with:
- loaded: List[str] loaded module names
- failed: List[Tuple[module, error]]
- total_parsers, total_validators, total_template_builders, total_region_processors
- wheel_results: optional, per-directory installation outcomes

Example:
```python
from marie.extract.registry.base import component_registry

summary = component_registry.initialize_from_config({
    "load_core_components": True,
    "external_component_modules": [
        "my_company.marie_ext.parsers",
        "my_company.marie_ext.validators",
    ],
    "wheel_directories": ["/opt/marie/wheels"],
    "watch_wheels": True,
})
```


## Logging and diagnostics

- Registration logs at info/debug with component name and class/type
- Missing lookups log a warning with available names
- External loading logs loaded/failed counts; strict mode raises on failure
- get_registry_info() provides a snapshot for UI/telemetry/health checks

## Error handling

- Duplicate validator or template builder registrations raise BadConfigSource
- External loading in strict mode raises BadConfigSource on any failure
- Wheel handling exceptions are logged per directory and included in results

## Best practices

- Use descriptive, stable names for components; they become public contract keys in configs
- Avoid name collisions; treat validator and template builder names as unique
- Keep region processor interfaces stable; prefer class-based processors for complex logic
- Use strict=True during CI to catch failing external component imports early
- Expose get_registry_info() in telemetry endpoints for visibility
- When developing dynamic components:
  - Test against a scratch wheel directory
  - Enable watch_wheels for local iteration
  - Use logging to verify registrations happened

## Common usage patterns

- Feature toggling: Register multiple parsers/validators and select by name via config
- Multi-tenant: Load tenant-specific component modules via initialize_external_components
- Runtime extension: Drop a new wheel in a watched directory to provision components without restart

