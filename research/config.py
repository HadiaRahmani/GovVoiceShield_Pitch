import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=Path(os.environ.get('GOVVOICE_WORK_DIR',str(ROOT.parents[1]/'work')))
CACHE=Path(os.environ.get('GOVVOICE_DATA_DIR',str(WORK/'audio-data')))
WORK.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
