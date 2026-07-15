(class_declaration
  name: (name) @name.definition.class) @definition.class

(function_definition
  name: (name) @name.definition.function) @definition.function

(method_declaration
  name: (name) @name.definition.function) @definition.function

(object_creation_expression
  [
    (qualified_name (name) @name.reference.class)
    (variable_name (name) @name.reference.class)
  ]) @reference.class

(function_call_expression
  function: [
    (qualified_name (name) @name.reference.call)
    (variable_name (name)) @name.reference.call
  ]) @reference.call

(scoped_call_expression
  name: (name) @name.reference.call) @reference.call

(member_call_expression
  name: (name) @name.reference.call) @reference.call


; Local modifications: variable-level definitions (locals, parameters,
; fields/properties) so downstream search can find variables in any scope.
(simple_parameter name: (variable_name (name) @name.definition.parameter)) @definition.parameter
(assignment_expression left: (variable_name (name) @name.definition.variable)) @definition.variable
(property_element (variable_name (name) @name.definition.property)) @definition.property
