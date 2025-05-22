_base_ = "base_dino_detr_ssod_voc.py"


classes = ('anterior teeth No FD', 'anterior teeth FD')
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/90_FD_train.json",
            img_prefix="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/images_all/",
            classes=classes,

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/90_FD_train.json",
            img_prefix="/netscratch/shehzadi/MICCAI25/FD-SOS-main/data/v1/images_all/",
            classes=classes,
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
            sample_ratio=[1, 4],
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
        unsup_weight=4.0,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=40000)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
