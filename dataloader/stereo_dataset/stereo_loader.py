from utils.logger import Logger as Log

from dataloader.stereo_dataset.stereodataset import build_stereoloader

def get_loader(cfg, mode, seed=None):
    """

    :param cfg:
    :param mode: val/ noval /test
    :return:
    """
    cfg_dataset = cfg["dataset"]

    if mode == 'test':
        test_loader = build_stereoloader(cfg_dataset, mode, seed=seed)
        Log.info("Get loader Done for Test...")
        return test_loader
    train_loader = build_stereoloader(cfg_dataset, "train", seed=seed)
    if mode == 'noval':
        Log.info("Get loader Done and there is no validation when training...")
        return train_loader, None
    if mode == 'val':
        val_loader = build_stereoloader(cfg_dataset, "val", seed=seed)
        Log.info("Get Both train and validation loader Done...")
        return train_loader, val_loader

    Log.info("You set a wrong mode")
    exit(1)