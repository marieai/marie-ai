"""
Avoid re-exporting the NER executor here.

Importing submodules under ``marie.executor.ner`` should not eagerly import the
full OCR and box-processing stack.
"""
