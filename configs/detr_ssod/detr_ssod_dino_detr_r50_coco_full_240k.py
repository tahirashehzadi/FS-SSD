
_base_ = "base_dino_detr_ssod_coco_full.py"

classes = ('anterior teeth No FD', 'anterior teeth FD')
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/90_FD_train.json",
            img_prefix="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/images_all/",

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/All-Datas/1Medical/Dentist/robo/teeth_robo/train_annotations_two.json",
            img_prefix="/netscratch/shehzadi/All-Datas/1Medical/Dentist/robo/teeth_robo/train/"

        ),
    ),
    test=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/40_FD_test.json",
            img_prefix="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/images_all/",
            classes=classes,
    ),
    val=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/20_FD_val.json",
            img_prefix="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/images_all/",
            classes=classes,

    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=80000)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/coco_full"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
