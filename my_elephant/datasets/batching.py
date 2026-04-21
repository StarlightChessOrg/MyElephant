"""训练用 batch 迭代与杂项数据工具。"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class ImageClass:
    """按类别存放图像路径（历史遗留，与象棋训练无直接关系）。"""

    name: str
    image_paths: list[str]

    def __str__(self) -> str:
        return f"{self.name}, {len(self.image_paths)} images"

    def __len__(self) -> int:
        return len(self.image_paths)


class Dataset:
    def __init__(self, data: np.ndarray, label: np.ndarray) -> None:
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        if data.shape[0] != label.shape[0]:
            raise ValueError("data 与 label 的第一维长度必须一致")
        self._num_examples = int(data.shape[0])

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def label(self) -> np.ndarray:
        return self._label

    def next_batch(self, batch_size: int, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray]:
        _ = shuffle  # 旧 API 保留；行为与最初实现一致（始终打乱）
        start = self._index_in_epoch
        # 与旧版一致：首轮与跨 epoch 时均打乱（旧代码未真正使用 shuffle 形参）
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(self._num_examples)
            np.random.shuffle(idx)
            self._data = self.data[idx]
            self._label = self.label[idx]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start : self._num_examples]
            label_rest_part = self.label[start : self._num_examples]
            idx0 = np.arange(self._num_examples)
            np.random.shuffle(idx0)
            self._data = self.data[idx0]
            self._label = self.label[idx0]

            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[0:end]
            label_new_part = self._label[0:end]
            return (
                np.concatenate((data_rest_part, data_new_part), axis=0),
                np.concatenate((label_rest_part, label_new_part), axis=0),
            )

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._data[start:end], self._label[start:end]


class ProgressBar:
    def __init__(self, worksum: int, info: str = "", auto_display: bool = True) -> None:
        self.worksum = worksum
        self.info = info
        self.finishsum = 0
        self.auto_display = auto_display

    def startjob(self) -> None:
        self.begin_time = time.time()

    def complete(self, num: int) -> None:
        self.gaptime = time.time() - self.begin_time
        self.finishsum += num
        if self.auto_display:
            self.display_progress_bar()

    def display_progress_bar(self) -> None:
        percent = self.finishsum * 100 / self.worksum
        eta_time = self.gaptime * 100 / (percent + 0.001) - self.gaptime
        bar_fill = int(percent // 2)
        strprogress = "[" + "=" * bar_fill + ">" + "-" * (50 - bar_fill) + "]"
        str_log = (
            f"{self.info} {percent:.2f} % {strprogress} "
            f"{self.finishsum}/{self.worksum}\t used:{self.gaptime:.0f}s eta:{eta_time:.0f} s"
        )
        sys.stdout.write("\r" + str_log)
        sys.stdout.flush()


def get_dataset(paths: str) -> list[ImageClass]:
    dataset: list[ImageClass] = []
    for path in paths.split(":"):
        path_exp = os.path.expanduser(path)
        classes = sorted(os.listdir(path_exp))
        for class_name in classes:
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
    return dataset


def split_dataset(
    dataset: list[ImageClass],
    split_ratio: float,
    mode: str,
) -> tuple[list[ImageClass], list[ImageClass]]:
    if mode == "SPLIT_CLASSES":
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes * split_ratio))
        train_set = [dataset[i] for i in class_indices[:split]]
        test_set = [dataset[i] for i in class_indices[split:]]
    elif mode == "SPLIT_IMAGES":
        train_set: list[ImageClass] = []
        test_set: list[ImageClass] = []
        min_nrof_images = 2
        for cls in dataset:
            paths = list(cls.image_paths)
            np.random.shuffle(paths)
            split = int(round(len(paths) * split_ratio))
            if split < min_nrof_images:
                continue
            train_set.append(ImageClass(cls.name, paths[:split]))
            test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError(f'Invalid train/test split mode "{mode}"')
    return train_set, test_set
