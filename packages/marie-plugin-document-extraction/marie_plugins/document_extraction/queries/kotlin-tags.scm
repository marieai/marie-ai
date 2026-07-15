; Definitions

(class_declaration
  (type_identifier) @name.definition.class) @definition.class

(function_declaration
  (simple_identifier) @name.definition.function) @definition.function

(object_declaration
  (type_identifier) @name.definition.object) @definition.object

; References

(call_expression
  [
    (simple_identifier) @name.reference.call
    (navigation_expression
      (navigation_suffix
        (simple_identifier) @name.reference.call))
  ]) @reference.call

(delegation_specifier
  [
    (user_type) @name.reference.type
    (constructor_invocation
      (user_type) @name.reference.type)
  ]) @reference.type


; Local modifications: variable-level definitions (locals, parameters,
; fields/properties) so downstream search can find variables in any scope.
(parameter (simple_identifier) @name.definition.parameter) @definition.parameter
(class_parameter (simple_identifier) @name.definition.property) @definition.property
(property_declaration (variable_declaration (simple_identifier) @name.definition.variable)) @definition.variable
