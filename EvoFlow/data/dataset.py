import os
import re
import hashlib
import random
import warnings
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import pandas as pd
from tqdm import tqdm

from data.cache import PreprocessedCache
from data.split import DatasetSplitManager


class ADLongitudinal3DDataset(Dataset):

    STRUCTURE_COLS = [
        'Hippocampus', 'Ventricles', 'Entorhinal',
        'WholeBrain', 'Fusiform', 'MidTemp',
    ]

    def __init__(
        self,
        data_root: str,
        clinical_csv: str,
        num_history: int = 3,
        volume_size: Tuple[int, int, int] = (96, 112, 96),
        mode: str = 'train',
        split_manager: DatasetSplitManager = None,
        normalize: bool = True,
        max_time_delta: float = 120.0,
        random_target: bool = True,
        cache_dir: str = None,
        use_cache: bool = True,
        preprocess_all: bool = False,
        use_patches: bool = False,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patch_stride: Tuple[int, int, int] = (32, 32, 32),
        num_workers_preprocess: int = 4,
    ):
        self.data_root = data_root
        self.num_history = num_history
        self.volume_size = volume_size
        self.mode = mode
        self.normalize = normalize
        self.max_time_delta = max_time_delta
        self.random_target = random_target and (mode == 'train')

        self.use_patches = use_patches
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.use_cache = use_cache
        cache_dir = self._resolve_cache_dir(cache_dir, data_root, volume_size)

        if cache_dir:
            print(f"  Cache directory: {cache_dir}")
        self.cache = PreprocessedCache(cache_dir, volume_size) if (use_cache and cache_dir) else None

        if split_manager:
            self.split_manager = split_manager
        else:
            self.split_manager = DatasetSplitManager(data_root)

        self.clinical_df = None
        self._clinical_index = {}
        if clinical_csv and os.path.exists(clinical_csv):
            self._load_and_index_clinical(clinical_csv)

        self.samples = self._collect_samples()
        self.all_nifti_paths = self._collect_all_paths()

        if preprocess_all and use_cache:
            self._preprocess_all_files()

        if self.use_patches:
            self._generate_patch_indices()

        print(f"Dataset [{mode}]: {len(self.samples)} samples")
        if self.use_patches:
            print(f"  Patch mode: {len(self.patch_samples)} patch samples")
        print(f"  Cache: {'enabled' if use_cache else 'disabled'}")

    # ------------------------------------------------------------------
    # Cache directory resolution
    # ------------------------------------------------------------------

    def _resolve_cache_dir(self, cache_dir, data_root, volume_size):
        if cache_dir is not None:
            return cache_dir

        hash_str = hashlib.md5(data_root.encode()).hexdigest()[:8]
        size_str = f'{volume_size[0]}x{volume_size[1]}x{volume_size[2]}'

        possible_dirs = [
            os.path.join(os.getcwd(), '.mri_cache', hash_str, size_str),
            os.path.join(os.path.expanduser('~'), '.mri_cache', hash_str, size_str),
        ]

        try:
            test_dir = os.path.join(data_root, '.cache')
            os.makedirs(test_dir, exist_ok=True)
            test_file = os.path.join(test_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return os.path.join(data_root, '.cache', size_str)
        except (OSError, PermissionError):
            for d in possible_dirs:
                try:
                    os.makedirs(d, exist_ok=True)
                    return d
                except (OSError, PermissionError):
                    continue
            print(f"  Warning: Cannot find writable cache directory, disabling cache")
            self.use_cache = False
            return None

    # ------------------------------------------------------------------
    # Clinical data
    # ------------------------------------------------------------------

    def _load_and_index_clinical(self, csv_path: str):
        df = pd.read_csv(csv_path)

        if 'PTID' in df.columns:
            df['subject_id'] = df['PTID'].astype(str)
        elif 'RID' in df.columns:
            df['subject_id'] = df['RID'].astype(str)

        if 'Month' not in df.columns and 'M' in df.columns:
            df['Month'] = df['M']

        if 'ICV' in df.columns:
            for col in self.STRUCTURE_COLS:
                if col in df.columns:
                    df[f'{col}_norm'] = df[col] / df['ICV'] * 100

        self.clinical_df = df

        for _, row in df.iterrows():
            sid = str(row.get('subject_id', ''))
            month = row.get('Month', 0)
            if pd.notna(month):
                month = int(round(float(month)))
                key = (sid, month)
                if key not in self._clinical_index:
                    self._clinical_index[key] = row.to_dict()

        print(f"  Clinical index: {len(self._clinical_index)} entries")

    def _get_clinical_row_fast(self, subject_id: str, month: float) -> Optional[dict]:
        month_int = int(round(month))

        key = (subject_id, month_int)
        if key in self._clinical_index:
            return self._clinical_index[key]

        for delta in [1, -1, 2, -2, 3, -3]:
            key = (subject_id, month_int + delta)
            if key in self._clinical_index:
                return self._clinical_index[key]

        return None

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def _collect_samples(self) -> List[dict]:
        samples = []
        allowed = self.split_manager.get_subjects(self.mode)

        for folder in os.listdir(self.data_root):
            path = os.path.join(self.data_root, folder)
            if not os.path.isdir(path):
                continue

            sid = self._parse_subject_id(folder)
            if not sid or sid not in allowed:
                continue

            visits = self._collect_visits(path, sid)
            if len(visits) < self.num_history + 1:
                continue

            visits.sort(key=lambda x: x['month'])

            for i in range(len(visits) - self.num_history):
                samples.append({
                    'subject_id': sid,
                    'history_visits': visits[i:i + self.num_history],
                    'possible_targets': visits[i + self.num_history:],
                })

        return samples

    def _parse_subject_id(self, folder_name: str) -> Optional[str]:
        if folder_name.startswith('.'):
            return None
        for pattern in [r'sub-(\d+_S_\d+)', r'sub-(\d+)', r'(\d+_S_\d+)']:
            match = re.match(pattern, folder_name)
            if match:
                return match.group(1)
        return folder_name

    def _collect_visits(self, subject_path: str, subject_id: str) -> List[dict]:
        visits = []
        visit_times = []

        for f in os.listdir(subject_path):
            fpath = os.path.join(subject_path, f)

            if f.endswith(('.nii', '.nii.gz')):
                dt = self._parse_datetime(f)
                if dt:
                    visit_times.append((dt, f, fpath))
            elif os.path.isdir(fpath):
                nifti = self._find_nifti(fpath)
                if nifti:
                    dt = self._parse_datetime(f)
                    if dt:
                        visit_times.append((dt, f, nifti))

        if not visit_times:
            return visits

        visit_times.sort(key=lambda x: x[0])
        baseline = visit_times[0][0]

        for dt, name, path in visit_times:
            months = (dt - baseline).days / 30.44
            visits.append({
                'visit_code': name,
                'month': round(months, 1),
                'path': path,
            })

        return visits

    def _parse_datetime(self, name: str) -> Optional[datetime]:
        patterns = [
            (r'ses-(\d{14})', '%Y%m%d%H%M%S'),
            (r'ses-(\d{8})(?!\d)', '%Y%m%d'),
            (r'(\d{8})', '%Y%m%d'),
        ]
        for pattern, fmt in patterns:
            match = re.search(pattern, name)
            if match:
                try:
                    dt = datetime.strptime(match.group(1), fmt)
                    if 1990 <= dt.year <= 2030:
                        return dt
                except ValueError:
                    continue
        return None

    def _find_nifti(self, session_path: str) -> Optional[str]:
        anat = os.path.join(session_path, 'anat')
        search_dirs = [anat, session_path] if os.path.exists(anat) else [session_path]

        exclude_keywords = [
            'mask', 'segment', 'label', 'seg.', '_seg',
            'aparc', 'aseg', 'parcellation', 'atlas',
            'wm.', 'gm.', 'csf.', 'tissue',
        ]

        priority_keywords = [
            'brain_1mm',
            't1w_brain',
            '_brain.',
            't1w',
        ]

        for d in search_dirs:
            if not os.path.exists(d):
                continue

            candidates = []
            for f in os.listdir(d):
                if not f.endswith(('.nii', '.nii.gz')):
                    continue

                f_lower = f.lower()

                if any(kw in f_lower for kw in exclude_keywords):
                    continue

                candidates.append(f)

            if not candidates:
                continue

            for priority_kw in priority_keywords:
                for f in candidates:
                    if priority_kw in f.lower():
                        return os.path.join(d, f)

            candidates.sort()
            return os.path.join(d, candidates[0])

        return None

    def _collect_all_paths(self) -> set:
        paths = set()
        for sample in self.samples:
            for v in sample['history_visits']:
                paths.add(v['path'])
            for v in sample['possible_targets']:
                paths.add(v['path'])
        return paths

    # ------------------------------------------------------------------
    # Volume I/O
    # ------------------------------------------------------------------

    def _preprocess_all_files(self):
        uncached = [p for p in self.all_nifti_paths if not self.cache.is_cached(p)]

        if not uncached:
            print("  All files already cached!")
            return

        print(f"  Preprocessing {len(uncached)} files...")
        for path in tqdm(uncached, desc="Caching"):
            try:
                vol = self._load_nifti_raw(path)
                vol = self._resize_volume(vol)
                vol = self._normalize_volume(vol)
                self.cache.save(path, vol)
            except Exception as e:
                warnings.warn(f"Failed to cache {path}: {e}")

        self.cache.save_meta_batch()
        print(f"  Cache saved to: {self.cache.cache_dir}")

    def _load_nifti_raw(self, path: str) -> np.ndarray:
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        return data

    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        if volume.shape == self.volume_size:
            return volume
        factors = [t / c for t, c in zip(self.volume_size, volume.shape)]
        return zoom(volume, factors, order=1)

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return volume
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        volume = (volume - p1) / (p99 - p1 + 1e-8)
        return volume * 2 - 1

    def _load_volume(self, path: str) -> np.ndarray:
        if self.cache and self.cache.is_cached(path):
            return self.cache.load(path)

        vol = self._load_nifti_raw(path)
        vol = self._resize_volume(vol)
        vol = self._normalize_volume(vol)

        if self.cache:
            self.cache.save(path, vol)

        return vol

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def _generate_patch_indices(self):
        self.patch_samples = []

        positions = []
        for d in range(0, self.volume_size[0] - self.patch_size[0] + 1, self.patch_stride[0]):
            for h in range(0, self.volume_size[1] - self.patch_size[1] + 1, self.patch_stride[1]):
                for w in range(0, self.volume_size[2] - self.patch_size[2] + 1, self.patch_stride[2]):
                    positions.append((d, h, w))

        self.patch_positions = positions

        for idx, sample in enumerate(self.samples):
            for pos in positions:
                self.patch_samples.append({
                    'sample_idx': idx,
                    'patch_pos': pos,
                })

    def _extract_patch(self, volume: np.ndarray, pos: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = pos
        pd, ph, pw = self.patch_size
        return volume[d:d+pd, h:h+ph, w:w+pw].copy()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_structure_features(self, subject_id: str, month: float) -> np.ndarray:
        features = np.zeros(len(self.STRUCTURE_COLS), dtype=np.float32)
        row = self._get_clinical_row_fast(subject_id, month)

        if row:
            for i, col in enumerate(self.STRUCTURE_COLS):
                val = row.get(f'{col}_norm') or row.get(col)
                if val is not None and pd.notna(val):
                    features[i] = float(val)

        return features

    def _compute_evolution_features(self, history_visits: List[dict], subject_id: str) -> Dict[str, torch.Tensor]:
        T = len(history_visits)
        n_struct = len(self.STRUCTURE_COLS)

        seq = np.zeros((T, n_struct), dtype=np.float32)
        months = np.zeros(T, dtype=np.float32)

        for t, v in enumerate(history_visits):
            seq[t] = self._get_structure_features(subject_id, v['month'])
            months[t] = v['month']

        rates = np.zeros(n_struct, dtype=np.float32)
        if T >= 2 and months[-1] > months[0]:
            dt = months[-1] - months[0]
            for i in range(n_struct):
                if np.any(seq[:, i] != 0):
                    rates[i] = (seq[-1, i] - seq[0, i]) / (dt + 1e-6)

        change = seq[-1] - seq[0]

        norm = np.array([0.5, 3.0, 0.2, 70.0, 1.0, 1.0], dtype=np.float32)
        norm = np.maximum(norm, 1e-6)

        return {
            'structure_sequence': torch.from_numpy(seq / norm),
            'evolution_rates': torch.from_numpy(rates / norm * 12),
            'cumulative_change': torch.from_numpy(change / norm),
        }

    def _get_clinical_features(self, subject_id: str, month: float) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._get_clinical_row_fast(subject_id, month)

        if not row:
            return torch.zeros(5), torch.zeros(3, dtype=torch.long)

        def safe_get(col, default=0):
            val = row.get(col)
            if val is not None and pd.notna(val) and not isinstance(val, str):
                return float(val)
            return default

        continuous = torch.tensor([
            safe_get('AGE', 70) / 100.0,
            safe_get('PTEDUCAT', 12) / 20.0,
            safe_get('MMSE', 25) / 30.0,
            safe_get('CDRSB', 0) / 18.0,
            safe_get('ADAS13', 20) / 85.0,
        ], dtype=torch.float32)

        sex = 1 if str(row.get('PTGENDER', 'Male')).lower() == 'female' else 0
        dx_str = str(row.get('DX', 'CN'))
        dx = 2 if ('Dementia' in dx_str or 'AD' in dx_str) else (1 if 'MCI' in dx_str else 0)
        apoe = 1 if safe_get('APOE4', 0) > 0 else 0

        return continuous, torch.tensor([sex, dx, apoe], dtype=torch.long)

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        if self.use_patches:
            return len(self.patch_samples)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_patches:
            return self._getitem_patch(idx)
        return self._getitem_full(idx)

    def _getitem_full(self, idx):
        sample = self.samples[idx]
        sid = sample['subject_id']
        history = sample['history_visits']

        hist_vols = []
        for v in history:
            vol = self._load_volume(v['path'])
            hist_vols.append(torch.from_numpy(vol).float().unsqueeze(0))
        hist_images = torch.stack(hist_vols)

        targets = sample['possible_targets']
        target_idx = random.randint(0, len(targets) - 1) if self.random_target and len(targets) > 1 else 0
        target_v = targets[target_idx]

        target_vol = self._load_volume(target_v['path'])
        target_image = torch.from_numpy(target_vol).float().unsqueeze(0)

        dt_months = target_v['month'] - history[-1]['month']

        evo = self._compute_evolution_features(history, sid)
        cont, cat = self._get_clinical_features(sid, history[-1]['month'])

        return {
            'history_images': hist_images,
            'target_image': target_image,
            'time_delta': torch.tensor([dt_months / self.max_time_delta]),
            'time_delta_months': float(dt_months),
            **evo,
            'continuous_features': cont,
            'categorical_features': cat,
            'subject_id': sid,
            'target_visit': target_v['visit_code'],
            'history_visits': [v['visit_code'] for v in history],
            'history_months': [v['month'] for v in history],
            'target_month': target_v['month'],
        }

    def _getitem_patch(self, idx):
        info = self.patch_samples[idx]
        sample = self.samples[info['sample_idx']]
        pos = info['patch_pos']
        sid = sample['subject_id']
        history = sample['history_visits']

        hist_patches = []
        for v in history:
            vol = self._load_volume(v['path'])
            patch = self._extract_patch(vol, pos)
            hist_patches.append(torch.from_numpy(patch).float().unsqueeze(0))
        hist_images = torch.stack(hist_patches)

        targets = sample['possible_targets']
        target_idx = random.randint(0, len(targets) - 1) if self.random_target and len(targets) > 1 else 0
        target_v = targets[target_idx]

        target_vol = self._load_volume(target_v['path'])
        target_patch = self._extract_patch(target_vol, pos)
        target_image = torch.from_numpy(target_patch).float().unsqueeze(0)

        dt_months = target_v['month'] - history[-1]['month']
        evo = self._compute_evolution_features(history, sid)
        cont, cat = self._get_clinical_features(sid, history[-1]['month'])

        return {
            'history_images': hist_images,
            'target_image': target_image,
            'time_delta': torch.tensor([dt_months / self.max_time_delta]),
            'time_delta_months': float(dt_months),
            **evo,
            'continuous_features': cont,
            'categorical_features': cat,
            'subject_id': sid,
            'patch_pos': pos,
            'sample_idx': info['sample_idx'],
            'target_visit': target_v['visit_code'],
            'history_visits': [v['visit_code'] for v in history],
            'history_months': [v['month'] for v in history],
            'target_month': target_v['month'],
        }
