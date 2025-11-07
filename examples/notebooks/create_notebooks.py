"""Helper script to create Jupyter notebooks from Python examples."""

import json
import re
from pathlib import Path

def create_notebook_structure():
    """Create basic notebook structure."""
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def markdown_cell(text):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n") if line.strip() or True]
    }

def code_cell(code):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

def parse_python_to_cells(py_file):
    """Parse Python file into notebook cells."""
    content = py_file.read_text()

    # Extract docstring as title
    docstring_match = re.match(r'"""(.*?)"""', content, re.DOTALL)
    cells = []

    if docstring_match:
        docstring = docstring_match.group(1).strip()
        title_line = docstring.split('\n')[0]
        rest_doc = '\n'.join(docstring.split('\n')[1:]).strip()

        cells.append(markdown_cell(f"# {title_line}\n\n{rest_doc}"))

        # Remove docstring from content
        content = content[docstring_match.end():].strip()

    # Split by print section headers
    sections = re.split(r'(print\("=" \* 70\).*?print\("=" \* 70\))', content, flags=re.DOTALL)

    current_code = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if this is a section header
        if 'print("=" * 70)' in section:
            # Save previous code
            if current_code:
                code_text = '\n'.join(current_code).strip()
                if code_text:
                    cells.append(code_cell(code_text))
                current_code = []

            # Extract section title
            title_match = re.search(r'print\("=" \* 70\)\s*print\("(.+?)"\)', section)
            if title_match:
                section_title = title_match.group(1)
                cells.append(markdown_cell(f"## {section_title}"))

            # Add the section code
            cells.append(code_cell(section))
        else:
            current_code.append(section)

    # Add remaining code
    if current_code:
        code_text = '\n'.join(current_code).strip()
        if code_text:
            cells.append(code_cell(code_text))

    return cells

# Create notebooks
notebooks = {
    "00_quickstart.ipynb": "../00_quickstart.py",
    "01_basic_operations.ipynb": "../01_basic_operations.py",
    "10_encoders_scalar.ipynb": "../10_encoders_scalar.py",
    "14_encoders_ngram.ipynb": "../14_encoders_ngram.py",
    "17_encoders_image.ipynb": "../17_encoders_image.py",
    "25_app_integration_patterns.ipynb": "../25_app_integration_patterns.py",
}

for nb_name, py_file in notebooks.items():
    print(f"Creating {nb_name}...")

    py_path = Path(__file__).parent / py_file
    if not py_path.exists():
        print(f"  Warning: {py_file} not found, skipping")
        continue

    nb = create_notebook_structure()
    nb["cells"] = parse_python_to_cells(py_path)

    nb_path = Path(__file__).parent / nb_name
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"  Created {nb_name} with {len(nb['cells'])} cells")

print("\nDone! All notebooks created.")
