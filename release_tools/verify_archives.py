#!/usr/bin/env python3
"""Verify distributable ZIP members and optionally extract without overwriting files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--extract-to', type=Path)
    args = parser.parse_args()
    total = 0
    for archive in args.archives:
        with zipfile.ZipFile(archive) as z:
            manifests = [n for n in z.namelist() if n.startswith('release_manifests/') and n.endswith('.json')]
            if len(manifests) != 1:
                raise ValueError(f'Expected one embedded manifest: {archive}')
            manifest = json.loads(z.read(manifests[0]))
            entries = manifest['files']
            expected_names = set(entries) | {manifests[0]}
            if set(z.namelist()) != expected_names or len(z.namelist()) != len(expected_names):
                raise ValueError(f'Unexpected or repeated ZIP member: {archive}')
            for name, expected in entries.items():
                parts = PurePosixPath(name)
                if parts.is_absolute() or '..' in parts.parts or '\\' in name:
                    raise ValueError(f'Unsafe member: {name}')
                destination = (args.extract_to / name).resolve() if args.extract_to else None
                if destination and not destination.is_relative_to(args.extract_to.resolve()):
                    raise ValueError(f'Unsafe destination: {name}')
                if destination and destination.exists():
                    if not destination.is_file() or digest(destination) != expected['sha256']:
                        raise FileExistsError(f'Refusing to replace a different file: {destination}')
                    output = None
                elif destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    output = destination.with_name(destination.name + '.extracting').open('xb')
                else:
                    output = None
                h, length = hashlib.sha256(), 0
                try:
                    with z.open(name) as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            h.update(chunk)
                            length += len(chunk)
                            if output:
                                output.write(chunk)
                finally:
                    if output:
                        output.close()
                if length != expected['bytes'] or h.hexdigest() != expected['sha256']:
                    raise ValueError(f'Integrity failure: {archive}: {name}')
                if output:
                    destination.with_name(destination.name + '.extracting').replace(destination)
                total += 1
            if args.extract_to:
                target = args.extract_to / manifests[0]
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = z.read(manifests[0])
                if target.exists() and target.read_bytes() != payload:
                    raise FileExistsError(target)
                if not target.exists():
                    target.write_bytes(payload)
        print(json.dumps({'archive': archive.name, 'members_verified': len(entries), 'passed': True}), flush=True)
    print(json.dumps({'passed': True, 'total_members_verified': total, 'extracted': args.extract_to is not None}))


if __name__ == '__main__':
    main()
