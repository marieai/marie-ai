(class_declaration
  name: (identifier) @name.definition.class) @definition.class

(method_declaration
  name: (identifier) @name.definition.method) @definition.method

(method_invocation
  name: (identifier) @name.reference.method
  arguments: (argument_list) @reference.call)

(interface_declaration
  name: (identifier) @name.definition.interface) @definition.interface

(type_list
  (type_identifier) @name.reference.interface) @reference.implementation

(object_creation_expression
  type: (type_identifier) @name.reference.class) @reference.class

(superclass (type_identifier) @name.reference.class) @reference.class


; Local modifications: variable-level definitions (locals, parameters,
; fields/properties) so downstream search can find variables in any scope.
(formal_parameter name: (identifier) @name.definition.parameter) @definition.parameter
(local_variable_declaration declarator: (variable_declarator name: (identifier) @name.definition.variable)) @definition.variable
(field_declaration declarator: (variable_declarator name: (identifier) @name.definition.property)) @definition.property
