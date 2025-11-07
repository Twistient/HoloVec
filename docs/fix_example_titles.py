#!/usr/bin/env python3
"""
Fix example titles by adding RST heading underlines.
"""
import glob
import re
from pathlib import Path

# Find all example Python files
examples_dir = Path(__file__).parent.parent / "examples"
example_files = sorted(examples_dir.glob("*.py"))

print(f"Found {len(example_files)} example files to fix")

for filepath in example_files:
    content = filepath.read_text()

    # Check if docstring starts with """ followed by a title
    match = re.match(r'"""(\n.*?)\n\n', content, re.DOTALL)
    if not match:
        print(f"Skipping {filepath.name}: no docstring found")
        continue

    docstring_first_part = match.group(1).strip()
    lines = docstring_first_part.split('\n')

    if not lines:
        print(f"Skipping {filepath.name}: empty docstring")
        continue

    # Get the title (first line)
    title = lines[0]

    # Check if already has underline
    if len(lines) > 1 and lines[1] and all(c == '=' for c in lines[1].strip()):
        print(f"Already fixed: {filepath.name}")
        continue

    # Create underline with same length as title
    underline = '=' * len(title)

    # Replace the docstring
    old_pattern = f'"""\n{title}\n'
    new_pattern = f'"""\n{title}\n{underline}\n'

    new_content = content.replace(old_pattern, new_pattern, 1)

    if new_content != content:
        filepath.write_text(new_content)
        print(f"Fixed: {filepath.name} - '{title}'")
    else:
        print(f"No change: {filepath.name}")

print("\nDone!")
