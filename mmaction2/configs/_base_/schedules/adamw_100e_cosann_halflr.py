# optimizer
optimizer = dict(
    type='AdamW', lr=0.00005, weight_decay=0.01)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', warmup="linear", warmup_iters=1000, warmup_ratio=1.0 / 10, min_lr_ratio=1e-5)
total_epochs = 100
