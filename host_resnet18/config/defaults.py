from config.config_node import ConfigNode

config = ConfigNode()

# system
config.system = ConfigNode()
config.system.device = 'cuda'
# dataset
config.dataset = ConfigNode()
config.dataset.dataroot = ''
config.dataset.dataset_csv = ''
config.dataset.num_classes = 100
# train
config.train = ConfigNode()
config.train.seed = 2022
config.train.start_epoch = 1
config.train.num_epochs = 100
config.train.batch_size = 128
config.train.lr = 0.001
config.train.lr_patience = 20
config.train.momentum = 0.9
config.train.weight_decay = 1e-4
config.train.print_freq = 10
config.train.es_patience = 20
# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.resize = 256
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = True
config.train.dataloader.shuffle = True
config.train.dataloader.num_workers = 6
config.train.dataloader.prefetch_factor = 5
# results
config.results = ConfigNode()
config.results.run_folder = 'runs'
# attack 
# Add the following statements when launching attacks.
# config.attack = ConfigNode()
# config.attack.wm_batchsize = 20
# config.attack.trigger_data = ''
# config.attack.trigger_csv = ''


def get_default_config():
    return config.clone()
