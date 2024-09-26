import os
import random

import torch
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import nibabel as nib
import cv2
import cloudinary.uploader
import mrcnn.model as modellib

import logging

from config import NETWORK, RESULT_DIR, CDN_DIR, ALPHA, COLORMAP, ROOT_DIR, CFG
from time import time
import colorsys
from skimage.measure import find_contours

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
        
    def predict_mrcnn(self, filepath, filename):
        mrcnn_weight = "/Net/logs/organ_cfg20230513T2033/mask_rcnn_organ_cfg_0025.h5"
        MRCNN_WEIGHT_ABSPATH = os.path.abspath(ROOT_DIR + mrcnn_weight)
        MODEL_ABSPATH = os.path.abspath(os.path.join(ROOT_DIR, "/Net/logs"))

        try: 
            self.logger.info("predict_mrcnn. Start predicting using mrcnn in {}".format(filepath))

            original_image = cv2.imread(filepath, cv2.IMREAD_COLOR)

            NETWORK_MRCNN = modellib.MaskRCNN(mode='inference', model_dir=MODEL_ABSPATH, config=CFG)
            NETWORK_MRCNN.load_weights(MRCNN_WEIGHT_ABSPATH, by_name=True)
            results = NETWORK_MRCNN.detect([original_image], verbose=1)

            classes_list = ['BG', 'Ginjal', 'Limpa', 'Hati']
            r = results[0]

            rois, masks, classes, confidences = self.filter_anomaly(r['rois'], r['masks'], r['class_ids'], r['scores'])

            save_result_dir = os.path.abspath(os.path.join(CDN_DIR, "result.png"))
            self.save_instances(original_image, rois, masks, classes, 
                                classes_list, save_result_dir, confidences)
            
            self.logger.info("roi {}, class_id {}".format(rois, classes))

            size_array = []
            for index in range(len(classes)):
                
                if r['class_ids'][index] == 1:
                    key = "Ginjal"
                elif r['class_ids'][index] == 2:
                    key = "Limpa"
                elif r['class_ids'][index] == 3:
                    key = "Hati"
                size_array.append((key, 
                                   int(r['rois'][index][3] - r['rois'][index][1]), 
                                   int(r['rois'][index][2] - r['rois'][index][0])))
            
            try: 
                self.logger.info("upload files to CDN")
                upload_path = os.path.abspath(os.path.join(CDN_DIR, save_result_dir))
                # self.logger.info(upload_path)
                response = cloudinary.uploader.upload(upload_path)
                url = response['secure_url']
                os.remove(save_result_dir)
                os.remove(filepath)
            except Exception as e:
                self.logger.error("Error during handling of png: {}".format(e))

            self.logger.info(size_array)
            return url, size_array
        
        except Exception as e:
            self.logger.error("Error during prediction process")
            raise e
        
    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def save_instances(self, image, boxes, masks, class_ids, class_names, save_dir,
                      scores=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Generate random colors
        colors = colors or self.random_colors(N)

        # masked_image = image.astype(np.uint32).copy()
        masked_image = image.copy()
        for i in range(N):
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                # masked_image = masked_image.astype(np.uint8).copy()
                cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, thickness=2)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                # masked_image = masked_image.astype(np.uint32)
                masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                pts = verts.reshape((-1, 1, 2)).astype(np.int32)
                # masked_image = masked_image.astype(np.uint8)
                cv2.polylines(masked_image, [pts], True, color, thickness=2)
                
            cv2.putText(masked_image, caption, (x1, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(save_dir, masked_image)

    def filter_anomaly(self, rois, masks, classes, confidence):
        indices_one = np.where(classes == 1)[0]
        indices_two = np.where(classes == 2)[0]
        indices_three = np.where(classes == 3)[0]
        
        sorted_indices_one = indices_one[np.argsort(confidence[indices_one])]
        sorted_indices_two = indices_two[np.argsort(confidence[indices_two])]
        sorted_indices_three = indices_three[np.argsort(confidence[indices_three])]
        
        top_two_ginjal = sorted_indices_one[-2:] if len(sorted_indices_one > 0) else []
        top_limpa = sorted_indices_two[-1:] if len(sorted_indices_two > 0) else []
        top_hati = sorted_indices_three[-1:] if len(sorted_indices_three > 0) else []
        
        combined_indices = list(top_two_ginjal) + list(top_limpa) + list(top_hati)
        
        filtered_classes = classes[combined_indices]
        filtered_confidences = confidence[combined_indices]
        filtered_masks = masks[:,:,combined_indices]
        filtered_rois = rois[combined_indices]
        
        return filtered_rois, filtered_masks, filtered_classes, filtered_confidences
