#!/bin/bash
number="${1:-"minor"}"
echo Updating $number

git fetch
git merge-base --is-ancestor origin/HEAD HEAD
if [ $? -ne 0 ]; then
  echo "local branch is not up to date with origin";
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "there are changes not committed";
  exit 1
fi

python -m hatch version $number
if [ $? -ne 0 ]; then
  echo "Update version with hatch failed";
  exit 1
fi
v=$(python -m hatch version)
git commit llama_api_server/__about__.py -m "Bump version to $v" --no-verify
git tag v$v
git push origin HEAD --tags
