"""Resolve preserved absolute provenance paths without rewriting archived JSON."""
from pathlib import Path


def resolve_archived_path(value, root):
    root = Path(root).resolve()
    value = Path(value)
    if not value.is_absolute():
        path = root / value
    else:
        parts = value.parts
        matches = [i for i, part in enumerate(parts) if part in {'results', 'data', 'models', 'manuscript'}]
        if not matches:
            raise ValueError(f'No recognized project-relative path in {value}')
        path = root.joinpath(*parts[matches[0]:])
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(path)
    return path
