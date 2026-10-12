"""Git clean filter: remove notebook outputs without changing the working copy.

Standard library only; run via the local Git filter configured by setup_git.py.
An empty placeholder notebook is passed through unchanged.
"""
import json
import sys


def clean(raw):
    if not raw.strip():
        return raw
    notebook = json.loads(raw.decode('utf-8-sig'))
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        # Widget state and timing metadata can carry local output/session details.
        cell.setdefault('metadata', {}).pop('execution', None)
        cell['metadata'].pop('ExecuteTime', None)
    metadata = notebook.setdefault('metadata', {})
    metadata.pop('widgets', None)
    metadata.pop('signature', None)
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + '\n').encode('utf-8')


if __name__ == '__main__':
    sys.stdout.buffer.write(clean(sys.stdin.buffer.read()))
