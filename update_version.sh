#!/bin/bash
number="${1:-"minor"}"
echo Updating $number
if [ -n "$(git status --porcelain)" ]; then
  echo "there are changes not committed";
  exit 1
fi

python -m hatch version $number
v=$(python -m hatch version)
git commit llama_api_server/__about__.py -m "Bump version to $v"
git tag v$v
