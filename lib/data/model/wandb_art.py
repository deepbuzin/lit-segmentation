from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetCfg:
    art_id: str
    root = None

    def register_art(self, run):
        art = run.use_artifact(self.art_id, type="dataset")
        self.root = Path(art.download())


