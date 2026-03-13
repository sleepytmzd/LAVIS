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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        valid_annotations = []
        invalid_count = 0

        for ann in self.annotation:
            image_rel = ann.get("image")
            caption = ann.get("state_description") or ann.get("caption")

            if image_rel and caption:
                valid_annotations.append(ann)
            else:
                invalid_count += 1
        
        print(f"Filtered out {invalid_count} invalid annotations. {len(valid_annotations)} valid annotations remain.")

        self.annotation = valid_annotations

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        caption = ann.get("state_description") or ann.get("caption")

        image = self.vis_processor(image)
        caption = self.text_processor(caption)

        return {"image": image, "text_input": caption}
