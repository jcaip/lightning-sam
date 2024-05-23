from box import Box

config = {
    "num_devices": 4,
    "batch_size": 4,
    "num_workers": 4,
    "num_epochs": 14,
    "eval_interval": 1,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        # "weight_decay": 0.1,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 500,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/home/jessecai/local/MODELS/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": False,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "sparse": True
    },
    "dataset": {
        "train": {
            "root_dir": "/home/jessecai/local/DATA/coco2017/train2017",
            "annotation_file": "/home/jessecai/local/DATA/coco2017/annotations/instances_train2017.json"
        },
        "val": {
            "root_dir": "/home/jessecai/local/DATA/coco2017/val2017",
            "annotation_file": "/home/jessecai/local/DATA/coco2017/annotations/instances_val2017.json"
        }
    }
}

cfg = Box(config)
