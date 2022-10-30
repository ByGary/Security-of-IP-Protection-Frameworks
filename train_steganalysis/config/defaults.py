from config.config_node import ConfigNode

config = ConfigNode()

config.seed = 2022
config.device = 'cuda'
config.run_folder = 'runs'
config.has_ckp = False
config.cover_path = ''
config.stego_path = ''
config.valid_cover_path = ''
config.valid_stego_path = ''
config.batch_size = 20
config.start_epoch = 1
config.num_epochs = 1000
config.print_freq = 10
config.resize = 256
config.train_size = 4000
config.val_size = 1000
config.test_size = 1000
config.p_rot = 0.1
config.p_hflip = 0.1
config.lr = 1e-3
config.ckp = ''


def get_default_config():
    return config.clone()


