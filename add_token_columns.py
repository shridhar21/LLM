"""Add blank historical token columns without estimating or overwriting other values.

Run: python add_token_columns.py emissions_reports/SV emissions_reports/SN emissions_reports/SC
An exact original-byte backup is kept beside each changed answers.csv.
"""
import argparse
import csv
import io
import os
from pathlib import Path


def add_columns(path):
    path = Path(path).resolve()
    original = path.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode('utf-8-sig'), newline=''))
    fields, rows = list(reader.fieldnames or []), list(reader)
    if 'question' not in fields or len(fields) != len(set(fields)) or any(None in row for row in rows):
        raise ValueError(f'{path}: invalid answer CSV schema; left unchanged')
    missing = [key for key in ('input_tokens','output_tokens') if key not in fields]
    if not missing:
        return False
    backup = path.with_name(path.name+'.before_token_columns.bak')
    # Never overwrite a previous backup.
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}')
    with backup.open('xb') as handle:
        handle.write(original)
    temporary = path.with_name(path.name+'.tokens.tmp')
    with temporary.open('x', encoding='utf-8-sig' if original.startswith(b'\xef\xbb\xbf') else 'utf-8', newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields+missing)
        writer.writeheader()
        writer.writerows(rows)
    with temporary.open(encoding='utf-8-sig',newline='') as handle:
        updated = list(csv.DictReader(handle))
    if [{key: row[key] for key in fields} for row in updated] != rows:
        raise ValueError(f'{path}: verification failed; original retained')
    os.replace(temporary,path)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+',type=Path)
    args=parser.parse_args()
    changed=0
    for root in args.folders:
        root=root.resolve()
        if not root.is_dir():parser.error(f'Folder does not exist: {root}')
        for path in sorted(root.rglob('answers.csv')):
            if not path.resolve().is_relative_to(root):
                raise ValueError(f'Answer file resolves outside selected folder: {path}')
            changed+=add_columns(path)
    print(f'Added token columns to {changed} answer CSVs; historical counts are blank. Original-byte backups retained.')


if __name__=='__main__':main()
