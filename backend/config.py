import os
import logging
import torch
import cloudinary
import numpy as np
import cv2
from dotenv import load_dotenv  
from Net import model

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

# COLORMAP = np.zeros((256, 1, 3), dtype=np.uint8)
# COLORMAP[:, :] = [0,0,0]
# COLORMAP[1, :] = [255, 0, 0]
# COLORMAP[2, :] = [255, 255, 0]
# COLORMAP[3, :] = [0, 255, 0]
# COLORMAP[4, :] = [0, 255, 255]
# COLORMAP[5, :] = [0, 0, 255]
# COLORMAP[6, :] = [255, 0, 255]
# COLORMAP[7, :] = [128, 0, 0]
# COLORMAP[8, :] = [128, 128, 0]
# COLORMAP[9, :] = [0, 128, 0]
# COLORMAP[10, :] = [0, 128, 128]
# COLORMAP[11, :] = [128, 0, 128]
# COLORMAP[12, :] = [128, 128, 128]
# COLORMAP[13, :] = [128, 128, 128]

COLORMAP = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
COLORMAP[0] = [0, 0, 0]

CLOUDINARY_CONFIG = cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)