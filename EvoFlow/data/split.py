import os
import re
import json
import random
import hashlib
from typing import Optional, Tuple


class DatasetSplitManager:

    def __init__(
        self,
        data_root: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42,
        split_file: str = None,
    ):
        self.data_root = data_root
        self.split_ratio = split_ratio
        self.random_seed = random_seed

        hash_str = hashlib.md5(data_root.encode()).hexdigest()[:8]
        self.split_file = split_file or f'dataset_split_{hash_str}.json'

        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self._load_or_create_split()

    def _load_or_create_split(self):
        if os.path.exists(self.split_file):
            print(f"Loading split from: {self.split_file}")
            with open(self.split_file, 'r') as f:
                data = json.load(f)
            self.train_subjects = set(data['train'])
            self.val_subjects = set(data['val'])
            self.test_subjects = set(data['test'])
        else:
            self._create_split()

    def _create_split(self):
        subjects = []
        for folder in os.listdir(self.data_root):
            if os.path.isdir(os.path.join(self.data_root, folder)) and not folder.startswith('.'):
                sid = self._parse_subject_id(folder)
                if sid:
                    subjects.append(sid)

        subjects = list(set(subjects))
        random.seed(self.random_seed)
        random.shuffle(subjects)

        n = len(subjects)
        n_train = int(n * self.split_ratio[0])
        n_val = int(n * self.split_ratio[1])

        self.train_subjects = set(subjects[:n_train])
        self.val_subjects = set(subjects[n_train:n_train + n_val])
        self.test_subjects = set(subjects[n_train + n_val:])

        with open(self.split_file, 'w') as f:
            json.dump({
                'train': list(self.train_subjects),
                'val': list(self.val_subjects),
                'test': list(self.test_subjects),
            }, f, indent=2)
        print(f"Saved split to: {self.split_file}")

    def _parse_subject_id(self, folder_name: str) -> Optional[str]:
        if folder_name.startswith('.'):
            return None
        match = re.match(r'sub-(\d+_S_\d+)', folder_name)
        if match:
            return match.group(1)
        match = re.match(r'sub-(\d+)', folder_name)
        if match:
            return match.group(1)
        match = re.match(r'(\d+_S_\d+)', folder_name)
        if match:
            return match.group(1)
        return folder_name

    def get_subjects(self, mode: str) -> set:
        if mode == 'train':
            return self.train_subjects
        elif mode == 'val':
            return self.val_subjects
        return self.test_subjects
