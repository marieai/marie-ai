#!/usr/bin/env bash

set -ex

export DEFAULT_BRANCH='master'
export BUILD_DIR=_build/dirhtml

echo "Building sphinx docs"
cd ./sphinx
#make clean && make markdown

SOURCEDIR=.
BUILDDIR=_build
sphinx-build -M markdown "$SOURCEDIR" "$BUILDDIR"

cd ..
echo "Copying sphinx api docs to docs folder"
api_docs_dir="./sphinx/_build/markdown"

if [ -d "$api_docs_dir" ]; then
    # replace current working directory withing the markdown files with the root of the docs folder
    # get parent directory of current working directory
    rm -rf ./docs/api
    mkdir -p ./docs/api
    cp -r $api_docs_dir/* ./docs/api
fi

category_file="./docs/api/_category_.json"

cat <<EOF > $category_file
{
  "label": "API",
  "position": 100,
  "link": {
    "type": "generated-index",
    "description": "API Documentation"
  }
}
EOF

echo "Building docosaurus site"
yarn build
