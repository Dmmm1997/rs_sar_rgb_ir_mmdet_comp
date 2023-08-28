_base_ = 'deformable-detr-refine_r50_16xb2-50e_comp.py'
model = dict(as_two_stage=True)
load_from = "pretrian_checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth"