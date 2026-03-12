"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.gameqa_datasets import GameQAImageTextDataset


@registry.register_builder("gameqa")
class GameQAPretrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = GameQAImageTextDataset
    eval_dataset_cls = GameQAImageTextDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gameqa/defaults.yaml",
    }
