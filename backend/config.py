import os
import logging
import torch
import cloudinary
import numpy as np
import cv2
from Net import model
from mrcnn.config_alt import Config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pred_dir = "/PredictResult"
receive_dir = '/ReceivedFile'
cd_dir = '/CloudinaryFile'
RESULT_DIR = os.path.abspath(ROOT_DIR + pred_dir)
UPLOAD_DIR = os.path.abspath(ROOT_DIR + receive_dir)
CDN_DIR = os.path.abspath(ROOT_DIR + cd_dir)

logger.info("Loading Model")
weight_dir = "/Net/50-0.539-0.644.pth"
state_dict_dir = os.path.abspath(ROOT_DIR + weight_dir)
NETWORK =  model.Net(training=False)
NETWORK.load_state_dict(torch.load(state_dict_dir, map_location=torch.device('cpu')))
NETWORK.eval()
logger.info("Model Finished Loading")

ALPHA = 0.5

COLORMAP = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
COLORMAP[0] = [0, 0, 0]

CLOUDINARY_CONFIG = cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)   

class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "organ_pred_cfg"
    # number of classes (background + Blue Marbles + Non Blue marbles)
    NUM_CLASSES = 1 + 3
    # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


CFG = PredictionConfig()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'