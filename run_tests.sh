#!/bin/sh
set -eu
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT HUP INT TERM
cp -R /app/. "$workdir/"
cd "$workdir"
git init -q
python -m unittest discover -s tests -v
