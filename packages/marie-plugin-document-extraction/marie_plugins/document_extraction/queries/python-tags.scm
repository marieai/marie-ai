(module (expression_statement (assignment left: (identifier) @name.definition.constant) @definition.constant))

; Local modification: the locked tree-sitter-python grammar emits module-level
; assignments without an expression_statement wrapper.
(module (assignment left: (identifier) @name.definition.constant) @definition.constant)

(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
      (identifier) @name.reference.call
      (attribute
        attribute: (identifier) @name.reference.call)
  ]) @reference.call


; Local modifications: variable-level definitions so downstream search can
; find locals, parameters, and instance attributes (stock tags queries only
; index navigation-level symbols).
(assignment left: (identifier) @name.definition.variable) @definition.variable

(parameters (identifier) @name.definition.parameter @definition.parameter)
(typed_parameter (identifier) @name.definition.parameter) @definition.parameter
(default_parameter name: (identifier) @name.definition.parameter) @definition.parameter
(typed_default_parameter name: (identifier) @name.definition.parameter) @definition.parameter

((assignment
  left: (attribute
    object: (identifier) @_self
    attribute: (identifier) @name.definition.property)) @definition.property
  (#eq? @_self "self"))
