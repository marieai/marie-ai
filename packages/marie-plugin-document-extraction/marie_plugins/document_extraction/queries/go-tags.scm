(
  (comment)* @doc
  .
  (function_declaration
    name: (identifier) @name.definition.function) @definition.function
  (#strip! @doc "^//\\s*")
  (#set-adjacent! @doc @definition.function)
)

(
  (comment)* @doc
  .
  (method_declaration
    name: (field_identifier) @name.definition.method) @definition.method
  (#strip! @doc "^//\\s*")
  (#set-adjacent! @doc @definition.method)
)

(call_expression
  function: [
    (identifier) @name.reference.call
    (parenthesized_expression (identifier) @name.reference.call)
    (selector_expression field: (field_identifier) @name.reference.call)
    (parenthesized_expression (selector_expression field: (field_identifier) @name.reference.call))
  ]) @reference.call

(type_spec
  name: (type_identifier) @name.definition.type) @definition.type

(type_identifier) @name.reference.type @reference.type

(package_clause "package" (package_identifier) @name.definition.module)

(type_declaration (type_spec name: (type_identifier) @name.definition.interface type: (interface_type)))

(type_declaration (type_spec name: (type_identifier) @name.definition.class type: (struct_type)))

(import_declaration (import_spec) @name.reference.module)

(var_declaration (var_spec name: (identifier) @name.definition.variable))

(const_declaration (const_spec name: (identifier) @name.definition.constant))


; Local modifications: variable-level definitions (locals, parameters,
; fields/properties) so downstream search can find variables in any scope.
(parameter_declaration name: (identifier) @name.definition.parameter) @definition.parameter
(short_var_declaration left: (expression_list (identifier) @name.definition.variable)) @definition.variable
(var_spec name: (identifier) @name.definition.variable) @definition.variable
(range_clause left: (expression_list (identifier) @name.definition.variable)) @definition.variable
(const_spec name: (identifier) @name.definition.constant) @definition.constant
(field_declaration name: (field_identifier) @name.definition.property) @definition.property
