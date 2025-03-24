#!/bin/bash

# Set default number of files to 10
nf=${1:-10}

# Set default number of rows to 1
nr=${2:-1}

# Find the n most recently updated *.out files and display their tail
find . -maxdepth 1 -type f -name '*.out' -printf '%T@ %p\n' | \
sort -n -r | \
head -n $nf | \
cut -d' ' -f2- | \
while read file; do
  echo "=== $file ==="
  tail "$file" -n $nr
  echo ""
done
