# Tags Query Attribution

The `*-tags.scm` files in this directory are vendored from the aider project (https://github.com/Aider-AI/aider), commit `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`, directories `aider/queries/tree-sitter-language-pack/` and `aider/queries/tree-sitter-languages/` (typescript). aider is licensed under the Apache License 2.0; these query files derive from the per-grammar tags queries maintained by the tree-sitter grammar communities.

Each file maps one language's syntax tree to standard `@definition.*`, `@name.*`, and `@reference.*` captures. To add a language, drop its `<language>-tags.scm` here — provider formats are derived from this directory.

Local modifications:

- `python-tags.scm`: a second `@definition.constant` pattern, because the locked `tree-sitter-language-pack==1.12.5` python grammar emits module-level assignments without an `expression_statement` wrapper (the upstream pattern compiles but never matches).
- All 13 fixture-backed languages (python, typescript, javascript, go, java, rust, c, cpp, csharp, ruby, php, kotlin, swift) gained variable-level definition patterns — `@definition.variable` (assignments/declarations in any scope), `@definition.parameter` (function/method parameters), `@definition.property` (fields, instance attributes) and, for go/rust, `@definition.constant` — so downstream search can find variables in any scope. Stock tags queries index only navigation-level symbols.
