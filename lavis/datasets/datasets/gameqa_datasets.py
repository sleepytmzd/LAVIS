"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset


class GameQAImageTextDataset(BaseDataset):
    """Image-text dataset for GameQA stage-1 pretraining."""

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_rel = ann.get("image")
        if not image_rel:
            return None

        image_path = os.path.join(self.vis_root, image_rel)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        caption = ann.get("state_description") or ann.get("caption")
        if caption is None:
            return None

        image = self.vis_processor(image)
        caption = self.text_processor(caption)

        return {"image": image, "text_input": caption}
