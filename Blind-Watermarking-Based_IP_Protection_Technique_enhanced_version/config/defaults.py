from config.config_node import ConfigNode

config = ConfigNode()

# system
config.system = ConfigNode()
config.system.device = 'cuda'
# data
config.dataset = ConfigNode()
config.dataset.dataroot = ''
config.dataset.dataset_csv = ''
config.dataset.num_classes = 100
# trigger
config.trigger = ConfigNode()
config.trigger.data = ''
config.trigger.csv = ''
# watermark
config.watermark = ConfigNode()
config.watermark.wm_root = ''
config.watermark.wm_csv = ''
config.watermark.wm_num = 480
config.watermark.wm_class = 4
config.watermark.wm_batchsize = 10
config.watermark.wm_resize = 256
# train
config.train = ConfigNode()
config.train.seed = 2022
config.train.pretrained_tech = True
config.train.fine_tuning = True
config.train.start_epoch = 1
config.train.num_epochs = 100
config.train.es_patience = 20
config.train.batchsize = 8
config.train.lr = 0.001
config.train.momentum = 0.9
config.train.weight_decay = 1e-4
config.train.loss_hyper_param = [3, 5, 1, 0.1]
config.train.print_freq = 10
# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.resize = 256
config.train.dataloader.drop_last = True
config.train.dataloader.shuffle = True
config.train.dataloader.pin_memory = True
config.train.dataloader.num_workers = 6
config.train.dataloader.prefetch_factor = 5
# runs
config.results = ConfigNode()
config.results.run_folder = 'runs'
# test
config.test = False


def get_default_config():
    return config.clone()


