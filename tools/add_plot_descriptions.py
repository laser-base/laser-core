#!/usr/bin/env python3
"""Insert markdown plot-description cells into Jupyter notebooks.

Driven by a JSON manifest of (notebook, marker, md_file) inserts. The script is
idempotent: it identifies an already-inserted description by matching the first
line of the markdown file against the next cell after the marker cell, so
re-running makes no changes.

Manifest format (paths are resolved relative to the manifest file's directory).
Each insert specifies the target cell by exactly ONE of `after_cell_containing`
(a unique substring of the code cell's source) or `after_cell_id` (the cell's
stable `id` field — use this when two code cells have identical source, so
no substring can disambiguate them):

    [
      {
        "notebook": "../examples/age_pyramid.ipynb",
        "inserts": [
          {
            "after_cell_containing": "plt.title(\"Age Distribution in Nigeria\")",
            "md_file": "age_pyramid_nigeria.md"
          },
          {
            "after_cell_id": "8f77666c",
            "md_file": "duplicate_cell_description.md"
          }
        ]
      }
    ]

Usage:
    python tools/add_plot_descriptions.py tools/plot_descriptions/config.json
    python tools/add_plot_descriptions.py tools/plot_descriptions/config.json --check
"""
import argparse
import json
import sys
from pathlib import Path


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def find_marker_cell(cells, marker):
    # Only match code cells. A description inserted as a markdown cell could
    # quote the marker string verbatim, which would make later --check passes
    # see it as ambiguous against the very prose it generated.
    matches = [
        i for i, c in enumerate(cells)
        if c.get("cell_type") == "code"
        and marker in "".join(c.get("source", []))
    ]
    if not matches:
        raise LookupError(f"no code cell contains marker {marker!r}")
    if len(matches) > 1:
        raise LookupError(f"marker {marker!r} is ambiguous (code cells {matches})")
    return matches[0]


def find_cell_by_id(cells, cell_id):
    # Cell IDs are stable across edits (assigned by Jupyter on cell creation),
    # so this disambiguates byte-for-byte identical code cells where no
    # substring marker can.
    matches = [i for i, c in enumerate(cells) if c.get("id") == cell_id]
    if not matches:
        raise LookupError(f"no cell with id {cell_id!r}")
    if len(matches) > 1:
        raise LookupError(f"cell id {cell_id!r} is not unique (cells {matches})")
    return matches[0]


def resolve_target_cell(cells, ins):
    has_marker = "after_cell_containing" in ins
    has_id = "after_cell_id" in ins
    if has_marker == has_id:
        raise ValueError("each insert must specify exactly one of after_cell_containing or after_cell_id")
    if has_marker:
        return find_marker_cell(cells, ins["after_cell_containing"])
    return find_cell_by_id(cells, ins["after_cell_id"])


def heading_of(text):
    return text.lstrip().splitlines()[0].strip()


def cell_heading(cell):
    if cell.get("cell_type") != "markdown":
        return None
    src = "".join(cell.get("source", []))
    if not src.strip():
        return None
    return heading_of(src)


def apply_to_notebook(nb_path: Path, inserts, base_dir: Path, *, write: bool = True):
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = data["cells"]
    plan = []
    for ins in inserts:
        idx = resolve_target_cell(cells, ins)
        text = (base_dir / ins["md_file"]).read_text(encoding="utf-8").rstrip() + "\n"
        plan.append((idx, text))

    changed = False
    # Apply in reverse index order so earlier inserts don't shift later positions.
    for idx, text in sorted(plan, key=lambda x: x[0], reverse=True):
        wanted = heading_of(text)
        next_idx = idx + 1

        if next_idx < len(cells) and cell_heading(cells[next_idx]) == wanted:
            existing = "".join(cells[next_idx].get("source", []))
            if existing != text:
                cells[next_idx]["source"] = text.splitlines(keepends=True)
                changed = True
            continue  # already present

        cells.insert(next_idx, md_cell(text))
        changed = True

    if changed and write:
        nb_path.write_text(
            json.dumps(data, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path, help="JSON manifest of inserts")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit with status 1 if any notebook would change (for CI)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    base_dir = manifest_path.parent
    entries = json.loads(manifest_path.read_text())

    any_change = False
    for entry in entries:
        nb_path = (base_dir / entry["notebook"]).resolve()
        # --check is a dry-run: report what would change without modifying notebooks.
        changed = apply_to_notebook(nb_path, entry["inserts"], base_dir, write=not args.check)
        print(f"{'CHANGED' if changed else 'ok     '} {nb_path}")
        any_change |= changed

    if args.check and any_change:
        sys.exit(1)


if __name__ == "__main__":
    main()
