"""Install a local notebook-output clean filter. Does not stage or commit files."""
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
command = ' '.join(shlex.quote(str(p).replace('\\', '/')) for p in [Path(sys.executable), root / 'scripts/notebook_git_clean.py'])
subprocess.run(['git', 'config', '--local', 'filter.notebook-clean.clean', command], cwd=root, check=True)
subprocess.run(['git', 'config', '--local', 'filter.notebook-clean.smudge', 'cat'], cwd=root, check=True)
subprocess.run(['git', 'config', '--local', 'filter.notebook-clean.required', 'true'], cwd=root, check=True)
print('Configured notebook-clean: staged notebook outputs are removed; local notebooks are preserved.')
