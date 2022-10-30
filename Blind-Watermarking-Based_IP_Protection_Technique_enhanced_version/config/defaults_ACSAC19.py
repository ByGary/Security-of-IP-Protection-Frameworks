from config.config_node import ConfigNode

config = ConfigNode()

config.dataset = ''
config.dataroot = ''
config.dataset_csv = ''
config.num_epochs = 100
config.batchsize = 100
config.wm_num = [480]
config.wm_batchsize = 20
config.lr = [0.001, 0.1]
config.hyper_parameters = [3, 5, 1, 0.1]
config.save_path = '/runs/'
config.seed = 32
config.pretrained = True
config.print_freq = 120


def get_default_config():
    return config.clone()


