import os
import tiffslide as openslide
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import histomicstk.preprocessing.color_deconvolution as htk
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import cv2
import numpy as np
import json
import argparse

def moving_average_smooth(contour, window=5):
    smoothed = np.copy(contour).astype(np.float32)
    for i in range(len(contour)):
        smoothed[i] = np.mean(contour[max(0, i - window):min(len(contour), i + window)], axis=0)
    return smoothed.astype(np.int32)

def StainDeconvolution(pil_image):
    pil_image = np.array(pil_image)
    stain_matrix = np.array([
        [0.650, 0.704, 0.286],
        [0.268, 0.570, 0.776],
        [0.714, 0.295, 0.636]
    ])
    deconvolved = htk.color_deconvolution(pil_image, stain_matrix)
    hem_channel = deconvolved.Stains[:, :, 0]
    eosin_channel = deconvolved.Stains[:, :, 1]
    dab_channel = deconvolved.Stains[:, :, 2]
    return hem_channel, eosin_channel, dab_channel

def contours_to_dsa(contours, slide_name="Slide_001", annotation_group="Region"):
    """
    Convert OpenCV contours to Digital Slide Archive (DSA) JSON annotation format.

    :param contours: List of contours from OpenCV (cv2.findContours)
    :param slide_name: Name of the slide for annotation
    :param annotation_group: Name of the annotation group
    :return: JSON formatted annotation data
    """
    dsa_annotations = {
        "name": slide_name,
        "elements": []
    }

    for contour in contours:
        # Convert contour to a list of (x, y) points
        points = [[int(p[0][0]), int(p[0][1]), 0] for p in contour]

        # Create DSA annotation entry
        annotation_entry = {
            "type": "polyline",  # "polyline" or "polygon" depending on need
            "lineColor": "rgb(255,0,0)",
            "fillColor": "rgba(255,0,0,0.3)",
            "lineWidth": 2,
            "closed": True,  # True for filled regions, False for open contours
            "points": points,
            "group": annotation_group
        }

        dsa_annotations["elements"].append(annotation_entry)

    return dsa_annotations

def processSlide(inputImage, outputAnnotations, thre_area_min = 10000, moving_window = 4, closing_size = 37):

    print(f">>>>>> segmentIhc: read svs file")
    slide = openslide.open_slide(inputImage)
    ds = int(slide.level_downsamples[slide.level_count -2])
    print(ds)

    loc = [0, 0]
    level = slide.level_count-2
    dim = slide.level_dimensions[slide.level_count-2]
    ds = int(slide.level_downsamples[slide.level_count -2])
    ld = slide.read_region(loc, level, dim)
    image_np = np.array(ld, dtype=np.uint8)
    if len(image_np.shape) == 3: 
        image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_np_gray = np.copy(image_np)

    hem_channel, eosin_channel, dab_channel = StainDeconvolution(ld)
    image_np_gray = dab_channel  

    images = []
    images.append(image_np)
    print(f">>>>>> segmentIhc: extract tissue mask")

    _, tissue_mask = cv2.threshold(image_np_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = 255-tissue_mask
    kernel = np.ones((closing_size,closing_size), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    y_indices, x_indices = np.where(tissue_mask == 255)  
    pixels = image_np[y_indices, x_indices, :]
    pixels_gray = image_np_gray[y_indices, x_indices]
    indexed_pixels = list(zip(x_indices, y_indices, pixels, pixels_gray))
    print(tissue_mask.shape)
    print(pixels.shape)
    print(pixels_gray.shape)
    print(indexed_pixels[0])
    grayscale_pixels_2d = pixels_gray.reshape(-1, 1)  
    print(f">>>>>> segmentIhc: segment chromogen")

    _, otsu_thresh = cv2.threshold(grayscale_pixels_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu_1d = 255-otsu_thresh.flatten()
    indexed_pixels = list(zip(x_indices, y_indices, pixels, pixels_gray, otsu_thresh))
    print(indexed_pixels[0])
    binary_mask = np.zeros_like(image_np_gray)  
    binary_mask[y_indices, x_indices] = otsu_1d 

    print(f">>>>>> segmentIhc: post processing")

    kernel = np.ones((closing_size,closing_size), np.uint8)  
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(opened)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= thre_area_min:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)  
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_contours = []
    for i in range(len(contours)):
        smoothed_contour = moving_average_smooth(contours[i], window=moving_window)
        smoothed_contours.append(smoothed_contour)
    filtered_mask_final = np.zeros_like(filtered_mask)
    cv2.drawContours(filtered_mask_final, smoothed_contours, -1, 255, -1)  
    scaled_contours = []
    for contour in smoothed_contours:
        scaled_contour = contour * ds  
        scaled_contours.append(scaled_contour)
    scaled_contours_smoothed = []
    for i in range(len(scaled_contours)):
        smoothed_contour = moving_average_smooth(scaled_contours[i], window=15)
        scaled_contours_smoothed.append(smoothed_contour)
    dim0 = slide.level_dimensions[0]
    filtered_mask_final_full = np.zeros((dim0[1], dim0[0]), dtype='uint8')
    image_np_display = np.copy(image_np)
    cv2.drawContours(image_np_display, scaled_contours_smoothed, -1, 255, 2)
    print(f"thre_area_min: {thre_area_min}")

    print(f">>>>>> segmentIhc: saving to output file")

    dsa_json = contours_to_dsa(scaled_contours_smoothed, slide_name=outputAnnotations.split('/')[-1].replace('.svs', ''))

    with open(outputAnnotations, "w") as f:
        json.dump(dsa_json, f, indent=4)

    print(f">>>>>> segmentIhc: DSA annotation file saved: {outputAnnotations}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--thre_area_min', type=int, default=10000)
    parser.add_argument('--moving_window', type=int, default=4)
    parser.add_argument('--closing_size', type=int, default=37)
    return parser.parse_args()

def main():
    args = get_args()
    processSlide(args.input_path, args.output_path, args.thre_area_min, args.moving_window, args.closing_size)

main()