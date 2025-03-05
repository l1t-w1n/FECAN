class Config:
    # Dataset
    train_hr_path = "path/to/your/train_hr_images"
    val_hr_path = "path/to/your/val_hr_images"
    patch_sizes = {"tiny": 64, "light": 64, "classical": 48}
    batch_size = 32
    num_workers = 8
    
    # Training
    model_type = "classical"  # tiny/light/classical
    scale_factor = 4
    lr = 5e-4
    min_lr = 1e-7
    betas = (0.9, 0.99)
    max_iter = 1600000
    save_interval = 5000
    
    # Loss weights
    l1_weight = 1.0
    freq_weight = 0.05
    
    # Augmentation
    rot_angles = [0, 90, 180, 270]
    hflip_prob = 0.5