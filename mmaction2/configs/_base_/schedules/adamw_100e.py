# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.01)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100
