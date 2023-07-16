from easydict import EasyDict as edict

env_settings = edict()

# env_settings.ROOT = '/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/'
# env_settings.DATA = '/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data'
# env_settings.EXPERIMENTS = '/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/experiments'
# env_settings.CUDA_VISIBLE_DEVICES = None
# env_settings.BIDS_PATH = None

env_settings.ROOT = '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction'
env_settings.DATA = '/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data'
env_settings.EXPERIMENTS = '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/'
env_settings.CUDA_VISIBLE_DEVICES = 3
env_settings.BIDS_PATH = '/data1/practical-sose23/castellvi/castellvi_prediction/bids'
env_settings.MASTER_LIST = '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/Castellvi_list_Final_Split_v2.xlsx'
