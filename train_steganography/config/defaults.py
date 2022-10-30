from config.config_node import ConfigNode

config = ConfigNode()

# system
config.system = ConfigNode()
config.system.device = 'cuda'
# train_on_mini
config.train_on_mini = ConfigNode()
config.train_on_mini.cover_data = ''
config.train_on_mini.secret_data = ''
config.train_on_mini.cover_csv = ''
config.train_on_mini.secret_csv = ''
# train
config.train = ConfigNode()
config.train.seed = 2022
config.train.has_ckp = False
config.train.start_epoch = 1
config.train.es_patience = 20
config.train.num_epochs = 100
config.train.batchsize = 20
config.train.lr = 0.001
config.train.print_freq = 10
# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.resize = 256
config.train.dataloader.drop_last = True
config.train.dataloader.shuffle = True
# results
config.results = ConfigNode()
config.results.run_folder = 'runs'
# Add the following statements when launching spoofing attack so as to match the YAML file.
# config.attack = ConfigNode()
# config.attack.cover_data = ''
# config.attack.cover_csv = ''
# config.attack.trigger_data = ''
# config.attack.trigger_csv = ''
# config.attack.wm_root = ''
# config.attack.wm_csv = ''
# config.attack.wm_batchsize = 20


def get_default_config():
    return config.clone()


