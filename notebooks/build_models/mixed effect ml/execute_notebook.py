"""Execute every notebook cell, preserve outputs, fail visibly on any error."""
import os
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

here = Path(__file__).resolve().parent
root = here.parents[2]
os.environ['JUPYTER_PATH'] = str(root / '.jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter/runtime')
path = here / 'next_mmse_merf_nan_CU_AD_nonAD.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=7200, kernel_name='merf-python', resources={'metadata': {'path': str(here)}},
                        on_cell_start=lambda cell, cell_index, **kw: print(f'Executing cell {cell_index}', flush=True))
out = root / 'results/merf_nan_CU_AD_nonAD'
out.mkdir(parents=True, exist_ok=True)
(out / 'execution_status.json').write_text(json.dumps({'real_data': 'running'}), encoding='utf-8')
try:
    client.execute()
except Exception as exc:
    (out / 'execution_status.json').write_text(json.dumps({'real_data': 'failed', 'error': repr(exc)}, ensure_ascii=False), encoding='utf-8')
    raise
finally:
    nbformat.write(notebook, path)
print('Notebook Run All completed', flush=True)
