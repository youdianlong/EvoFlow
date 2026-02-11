import json
import hashlib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict


class PreprocessedCache:

    def __init__(self, cache_dir: str, volume_size: Tuple[int, int, int]):
        self.cache_dir = Path(cache_dir)
        self.volume_size = volume_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.meta_file = self.cache_dir / 'cache_meta.json'
        self.meta = self._load_meta()

        self._path_to_cache: Dict[str, Path] = {}
        self._build_path_mapping()

    def _load_meta(self) -> dict:
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                return json.load(f)
        return {'volume_size': list(self.volume_size), 'files': {}}

    def _save_meta(self):
        with open(self.meta_file, 'w') as f:
            json.dump(self.meta, f, indent=2)

    def _build_path_mapping(self):
        for key, info in self.meta.get('files', {}).items():
            source_path = info.get('source', '')
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                self._path_to_cache[source_path] = cache_file

    def _get_cache_key(self, nifti_path: str) -> str:
        key_str = f"{nifti_path}_{self.volume_size}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get_cache_path(self, nifti_path: str) -> Path:
        if nifti_path in self._path_to_cache:
            return self._path_to_cache[nifti_path]
        key = self._get_cache_key(nifti_path)
        return self.cache_dir / f"{key}.npy"

    def is_cached(self, nifti_path: str) -> bool:
        return nifti_path in self._path_to_cache

    def save(self, nifti_path: str, volume: np.ndarray):
        key = self._get_cache_key(nifti_path)
        cache_path = self.cache_dir / f"{key}.npy"
        np.save(cache_path, volume.astype(np.float16))

        self._path_to_cache[nifti_path] = cache_path

        self.meta['files'][key] = {
            'source': nifti_path,
            'cached_at': datetime.now().isoformat()
        }

    def save_meta_batch(self):
        self._save_meta()

    def load(self, nifti_path: str) -> np.ndarray:
        cache_path = self._path_to_cache.get(nifti_path)
        if cache_path is None:
            raise FileNotFoundError(f"Cache not found for: {nifti_path}")
        return np.load(cache_path).astype(np.float32)
