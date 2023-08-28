# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset
import os.path as osp
from mmengine.fileio import list_from_file
from typing import List


@DATASETS.register_module()
class SAR_RS_2023(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes': ('car',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228),]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metainfo['dataset_type'] = None

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.tif')
            xml_path = osp.join(self.sub_data_root, self.ann_subdir,
                                f'{img_id}.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list