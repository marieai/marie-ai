(struct_specifier name: (type_identifier) @name.definition.class body:(_)) @definition.class

(declaration type: (union_specifier name: (type_identifier) @name.definition.class)) @definition.class

(function_declarator declarator: (identifier) @name.definition.function) @definition.function

(type_definition declarator: (type_identifier) @name.definition.type) @definition.type

(enum_specifier name: (type_identifier) @name.definition.type) @definition.type


; Local modifications: variable-level definitions (locals, parameters,
; fields/properties) so downstream search can find variables in any scope.
(parameter_declaration declarator: (identifier) @name.definition.parameter) @definition.parameter
(parameter_declaration declarator: (pointer_declarator declarator: (identifier) @name.definition.parameter)) @definition.parameter
(declaration declarator: (init_declarator declarator: (identifier) @name.definition.variable)) @definition.variable
(declaration declarator: (identifier) @name.definition.variable) @definition.variable
(field_declaration declarator: (field_identifier) @name.definition.property) @definition.property
