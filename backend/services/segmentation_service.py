import os

import torch
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import nibabel as nib
import cv2
import cloudinary.uploader

import logging

from config import NETWORK, RESULT_DIR, CDN_DIR, ALPHA, COLORMAP
from time import time

class SegmentationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.upper = 350
        self.lower = -self.upper
        self.down_scale = 0.5
        self.size = 48
        self.slice_thickness = 3

    def generate_png(self, ct_array_original, pred_seg, aspect_ratio, new_dims, filename):
        png_list = []

        colormap = COLORMAP

        ct_slice = ct_array_original[ct_array_original.shape[0]//2, :, :]
        ct_slice = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        ct_slice = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)

        seg_slice = pred_seg[pred_seg.shape[0]//2, :, :]
        seg_slice = cv2.normalize(seg_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # seg_slice = cv2.applyColorMap(seg_slice, cv2.COLORMAP_JET)
        seg_slice = cv2.applyColorMap(seg_slice, colormap)
        
        overlay_coronal = cv2.addWeighted(ct_slice, 1-ALPHA, seg_slice, ALPHA, 0)
        overlay_coronal = cv2.resize(overlay_coronal, (new_dims[2], new_dims[1]))

        name = CDN_DIR+'/'+filename+'_coronal'+'.png' 
        cv2.imwrite(name, overlay_coronal)
        png_list.append(name)
        
        ct_slice = ct_array_original[:, ct_array_original.shape[1]//2, :]
        ct_slice = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        ct_slice = cv2.flip(ct_slice, 0)
        ct_slice = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)

        seg_slice = pred_seg[:, ct_array_original.shape[1]//2, :]
        seg_slice = cv2.normalize(seg_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        seg_slice = cv2.flip(seg_slice, 0)
        # seg_slice = cv2.applyColorMap(seg_slice, cv2.COLORMAP_JET)
        seg_slice = cv2.applyColorMap(seg_slice, colormap)
        
        overlay_sagittal = cv2.addWeighted(ct_slice, 1-ALPHA, seg_slice, ALPHA, 0)
        overlay_sagittal = cv2.resize(overlay_sagittal, (new_dims[2], new_dims[0]))

        name = CDN_DIR+'/'+filename+'_sagittal'+'.png'
        cv2.imwrite(name, overlay_sagittal)
        png_list.append(name)
        
        ct_slice = ct_array_original[:, :, ct_array_original.shape[2]//2]
        ct_slice = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        ct_slice = cv2.flip(ct_slice, 0)
        ct_slice = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)

        seg_slice = pred_seg[:, :, ct_array_original.shape[2]//2]
        seg_slice = cv2.normalize(seg_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        seg_slice = cv2.flip(seg_slice, 0)
        # seg_slice = cv2.applyColorMap(seg_slice, cv2.COLORMAP_JET)
        seg_slice = cv2.applyColorMap(seg_slice, colormap)
        
        overlay_axial = cv2.addWeighted(ct_slice, 1-ALPHA, seg_slice, ALPHA, 0)
        overlay_axial = cv2.resize(overlay_axial, (new_dims[1], new_dims[0]))

        name = CDN_DIR+'/'+filename+'_axial'+'.png'
        cv2.imwrite(name, overlay_axial)
        png_list.append(name)

        return png_list
    
    def send_png(self, png_list):
        url_list = []
        for file_png in png_list:
            try: 
                response = cloudinary.uploader.upload(file_png)
                url_list.append(response['secure_url'])
                os.remove(file_png)
            except Exception as e:
                self.logger.error("Error during handling of png: {}".format(e))
            
        return url_list


    def predict_nifti(self, filepath, filename):
        try:
            self.logger.info("predict_nifti. Start predicting nifti file in {}".format(filepath))
            ct = sitk.ReadImage(filepath, sitk.sitkInt16)

            ct_nib = nib.load(filepath)
            pix_dim = ct_nib.header["pixdim"][1:4]
            aspect_ratio = [pix_dim[1]/pix_dim[2],pix_dim[0]/pix_dim[2],pix_dim[0]/pix_dim[1]]

            ct_array_original = sitk.GetArrayFromImage(ct)

            original_shape = ct_array_original.shape

            new_dims = np.multiply(ct_nib.get_fdata().shape, pix_dim)
            new_dims = (round(new_dims[0]),round(new_dims[1]),round(new_dims[2]))

            ct_array = ndimage.zoom(ct_array_original, (ct.GetSpacing()[-1] / self.slice_thickness, self.down_scale, self.down_scale), order=3)

            # slice and sample in the axial direction
            flag  =  False
            start_slice = 0
            end_slice = start_slice + self.size - 1
            ct_array_list = []

            while end_slice <= ct_array.shape[0] - 1:
                ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

                start_slice = end_slice + 1
                end_slice = start_slice + self.size - 1

            # When it is not divisible, take the last block in reverse
            if end_slice is not ct_array.shape[0] - 1:
                flag = True
                count = ct_array.shape[0] - start_slice
                ct_array_list.append(ct_array[-self.size:, :, :])

            outputs_list = []
            with torch.no_grad():
                for ct_array in ct_array_list:

                    ct_tensor = torch.FloatTensor(ct_array)
                    ct_tensor = ct_tensor.unsqueeze(dim=0)
                    ct_tensor = ct_tensor.unsqueeze(dim=0)

                    outputs = NETWORK(ct_tensor)
                    outputs = outputs.squeeze()

                    outputs_list.append(outputs.cpu().detach().numpy())
                    del outputs

            # Start splicing results after execution
            pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
            if flag is False:
                pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
            else:
                pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)

            pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0)
            pred_seg = F.interpolate(pred_seg, original_shape, mode='trilinear').squeeze().detach().numpy()
            pred_seg = np.argmax(pred_seg, axis=0)
            pred_seg = np.round(pred_seg).astype(np.uint8)

            png_list = self.generate_png(ct_array_original, pred_seg, aspect_ratio, new_dims, filename)
            url_list = self.send_png(png_list)

            self.logger.info("URL List: {}".format(url_list))

            pred_seg = sitk.GetImageFromArray(pred_seg)

            pred_seg.SetDirection(ct.GetDirection())
            pred_seg.SetOrigin(ct.GetOrigin())
            pred_seg.SetSpacing(ct.GetSpacing())

            file_destination = os.path.abspath(os.path.join(RESULT_DIR, "result.nii.gz"))

            sitk.WriteImage(pred_seg, file_destination)
            self.logger.info("predict_nifti. Predicted File Saved in {}".format(file_destination))
            del pred_seg

            return file_destination, url_list
        
        except Exception as e:
            self.logger.error("Error during prediction process")
            raise e

    
