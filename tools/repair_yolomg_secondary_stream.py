#!/usr/bin/env python
import argparse
from pathlib import Path

import cv2
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    return parser.parse_args()


def read_paths(path):
    return [Path(line.strip()) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_paths(path, paths):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text('\n'.join(str(item) for item in paths) + '\n', encoding='utf-8')
    temporary.replace(path)


def repair_split(data, split):
    primary_list = Path(data[split])
    secondary_list = Path(data[f'{split}2'])
    primary_paths = read_paths(primary_list)
    secondary_paths = read_paths(secondary_list)
    secondary_by_name = {path.name: path for path in secondary_paths}
    if len(secondary_by_name) != len(secondary_paths):
        raise ValueError(f'{secondary_list} contains duplicate basenames')

    repaired_paths = []
    created = 0
    for primary_path in primary_paths:
        secondary_path = secondary_by_name.get(primary_path.name)
        if secondary_path is None or not secondary_path.is_file():
            secondary_path = Path(str(primary_path).replace('\\images\\', '\\images2\\'))
            if secondary_path == primary_path:
                raise ValueError(f'Cannot derive secondary path from {primary_path}')
            image = cv2.imread(str(primary_path))
            if image is None:
                raise FileNotFoundError(primary_path)
            secondary_path.parent.mkdir(parents=True, exist_ok=True)
            zero_motion = image[:, :, 0] * 0
            if not cv2.imwrite(str(secondary_path), zero_motion):
                raise OSError(f'Failed to write {secondary_path}')
            created += 1
        repaired_paths.append(secondary_path)

    write_paths(secondary_list, repaired_paths)
    cache_path = secondary_list.with_suffix('.cache')
    if cache_path.exists():
        cache_path.unlink()
    print(f'{split}: primary={len(primary_paths)} secondary={len(repaired_paths)} created={created}')


def main():
    args = parse_args()
    data = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    for split in ('train', 'val', 'test'):
        repair_split(data, split)


if __name__ == '__main__':
    main()
