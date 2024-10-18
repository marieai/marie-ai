#!/bin/bash

for dir in marie/*/; do
  bump-pydantic "$dir"
done