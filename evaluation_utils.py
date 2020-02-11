import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import json
from shapely.geometry import Polygon
import image_slicer

# DeepLab code:
# taken from https://gluon-cv.mxnet.io/build/examples_segmentation/demo_deeplab.html
# dataset description https://groups.csail.mit.edu/vision/datasets/ADE20K/, https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/ade20k/segmentation.py
# deeplab code https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/deeplabv3.py

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
# using cpu
ctx = mx.cpu(0)

# OCR code
import pytesseract
from pytesseract import Output


# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Path('C:/Program Files\ Tesseract-OCR\ tesseract').as_posix()


# Scene recognition imports
from Keras_VGG16_places365.vgg16_places_365 import VGG16_Places365
from cv2 import resize


# Pre-processing functions for tesseract (OCR)
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def prepare_dataset(path_image_folder, model_type):
    list_image = []
    # Go through the directory
    list_image_names = []
    pathlist = Path(path_image_folder).glob('**/*.*')
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        list_image_names.append(path.stem)
        # print(path_in_str)
        
        # check for which model the data will be used and pre-process accordingly.
        if model_type == 'deeplab':
            img = image.imread(path_in_str)
            img = test_transform(img, ctx)
        
        elif model_type == 'OCR':
            img = cv2.imread(path_in_str)
            img = get_grayscale(img)
            img = thresholding(img)
        
        elif model_type == 'vgg_places365':
            img = Image.open(path_in_str)
            img = np.array(img, dtype=np.uint8)
            img = resize(img, (224, 224))
            img = np.expand_dims(img, 0)

        list_image.append(img)

    return list_image, list_image_names


def load_model(model_type):
    if model_type == 'deeplab':
        model = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True)
    return model


def get_predictions(input_data, model_type, loaded_model=''):
    
    if model_type == 'vgg_places365':
        file_name = Path('Keras_VGG16_places365/categories_places365.txt')
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
    
    output = []
    nb_im = 0
    for img in input_data:
        nb_im += 1
        print("Currently treating image number ", nb_im)
        if model_type == 'deeplab':
            pred = loaded_model.predict(img)
            # Check what the outputs of predict are: is it probability? does it depend on the class or is it class-agnostic?
            idx_labels = mx.nd.squeeze(mx.nd.argmax(pred, 1)).asnumpy()
            # Get the probabilities
            for pixel_h in range(0, pred.shape[2]):
            	for pixel_v in range(0, pred.shape[3]):
            		pred[0, :, pixel_h, pixel_v] = pred[0, :, pixel_h, pixel_v].softmax()
            # Decide later whether we make it into actual labels.
            output.append((pred.asnumpy(), idx_labels)) # this gives both the prediction "confidence" and the final label (in idx).
            #output.append(pred)
            # Maybe we'll need to use predictions probabilityes and not only labels.
        elif model_type == 'OCR': 
            pred = pytesseract.image_to_data(img, output_type=Output.DICT)
            prediction_list = []
            # Get the boundix boxes around the words
            n_boxes = len(pred['text'])
            for i in range(n_boxes):
                # if int(pred['conf'][i]) > 60: to filter per confidence ! see later!
                (x, y, w, h) = (pred['left'][i], pred['top'][i], pred['width'][i], pred['height'][i])
                prediction_list.append((pred['text'][i], (x, y, w, h)))
            output.append(prediction_list)
        elif model_type == 'vgg_places365':
            model = VGG16_Places365(weights='places')
            #predictions_to_return = 5
            preds = model.predict(img)[0]
            top_preds = np.argsort(preds)[::-1]#[0:predictions_to_return]
            top_preds_score = [preds[i] for i in top_preds]
            prediction_list = []
            for i in range(0, len(top_preds)):
                prediction_list.append((classes[top_preds[i]], top_preds_score[i]))
            output.append(prediction_list)

    return output
        
    # Add post processing to reshape the image / bounding boxes /  to original size
    # It seems there's no need for that becaause only the scene recognition model needs resizing.


# Evaluation

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def create_image_mask(image, polygon):
    # From an image and the coordinates of a polygon, create a binary matrix (1 when in the polygon, 0 otherwise).
    nx, ny = im.size
    img = Image.new("L", [nx, ny], 0)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    mask = np.array(img)
    return mask

def combine_image_masks(list_masks):
    new_mask = list_masks[0]
    if len(list_masks) > 1:
        for m in range(1, len(list_masks)):
            new_mask = np.add(new_mask, list_masks[m])
    # Convert back to binary matrix
    new_mask = np.where(new_mask > 0, 1, 0)
    return new_mask

def compute_mask_accuracy(GT_mask, pred_mask):
    # Get total number of pixels for the GT mask
    n_pixel_total = (GT_mask == 1).sum()
    # Get number of overlapping pixels with the GT mask
    mask_diff = np.subtract(GT_mask, pred_mask)
    n_incorrect_pixel = (mask_diff == 1).sum()
    acc = (n_pixel_total - n_incorrect_pixel) / n_pixel_total
    return acc
    
def evaluate_privacy(priv_elem_GT, predictions, image):
        
    # Go through each private element in predictions
    print("TODO: read predictions.")
    list_poly_pred = []

    
    dict_iou = {}
    
    ### Compute overlap
    for instance_priv in priv_elem_GT:
            
        for instance_priv_poly in priv_elem_GT[instance_priv]:
            # Compute IoU for each prediction
            a = Polygon(instance_priv_poly)
            
            iou_preds = []
            # Get the max IoU with all the predicted things.
            for poly in list_poly_pred:
                b = Polygon(poly)
                iou_preds.append(a.intersection(b).area / a.union(b).area)
            
            if instance_priv in dict_iou:
                dict_iou[instance_priv].append(max(iou_preds))
            else:
                dict_iou[instance_priv] = max(iou_preds)
                
    
    ### Compute pixel-wise privacy-element -wise accuracy  
    dict_pixel_perf = {}
    # Get the mask for the predictions
    list_mask_pred = []
    for poly in list_poly_pred:
        list_mask_pred.append(create_image_mask(image, poly))
    pred_mask = combine_image_masks(list_mask_pred)
                         
    # Go through each private element of each category
    for instance_priv in priv_elem_GT:
        # Create the binary mask for the private element:
        list_masks_GT = []
        for instance_priv_poly in priv_elem_GT[instance_priv]:
            # Compare each pixel of the private element
            list_masks_GT.append(create_image_mask(image, instance_priv_poly))
        GT_mask = combine_image_masks(list_masks_GT)
    
        # Check 1 if obfuscated, 0 otherwise. 
        # Get accuracy. (number 1 / total number pixels)
        acc = compute_mask_accuracy(GT_mask, pred_mask)
        dict_pixel_perf[instance_priv] = acc
    return dict_iou, dict_pixel_perf

                         
def GT_annotation_to_polygon_dict(ground_truth):
    priv_elem_GT = {}
    for private_elem in ground_truth['attributes']:
        name = private_elem['attr_id']
        list_polygons = private_elem['polygons']
        # Reshape the polygons into a readable format.
        readable_poly = []
        for poly in list_polygons:
            p = []
            for x, y in pairwise(poly):
                p.append((x, y))
            readable_poly.append(p)  
        if name in priv_elem_GT:
            priv_elem_GT[name].append(readable_poly)
        else:
            priv_elem_GT[name] = readable_poly
    return priv_elem_GT
                         
def segment_array(array, segmentation_size):
    im_w = array.shape[1] 
    im_h = array.shape[0]
    print("TODO: check the validity / col/row")
    columns, rows = image_slicer.calc_columns_rows(segmentation_size)
    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))
    segments = []
    for pos_y in range(0, im_h - rows, tile_h): # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w): # as above.
            #area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            segments.append[array[pos_x:(pos_x + tile_w)][pos_y:(pos_y + tile_h)]] 
            print("TODO: check the sizes")
    return segments
                         
def evaluate_instance(ground_truth, predictions, segmentation_size):
    ### Compute the image masks
    # For the GT
    list_masks_GT = []
    for instance_priv in priv_elem_GT:
        # Create the binary mask for the private element:
        for instance_priv_poly in priv_elem_GT[instance_priv]:
            # Compare each pixel of the private element
            list_masks_GT.append(create_image_mask(image, instance_priv_poly))
    GT_mask = combine_image_masks(list_masks_GT)
    # For the predictions
    list_mask_pred = []
    for poly in list_poly_pred:
        list_mask_pred.append(create_image_mask(image, poly))
    pred_mask = combine_image_masks(list_mask_pred)                     
                         
    ### Segment the masks
    segments_GT = segment_array(GT_mask, segmentation_size)
    segments_pred = segment_array(pred_mask, segmentation_size)
    segment_size = segments_GT[0].shape[0] * segments_GT[0].shape[1]    
    threshold_pixels = segment_size / 2
    ### Compute numbers TP, TN, FP, FN
    dict_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for segment_GT, segment_pred in zip(segments_GT, segments_pred):
        print("Check whether these rules are making sense.")
        print("For now, for each segment, we say that if more than half of the pixels are set to 1, then the segment is set at 1.")
        # Get the number of 1s in each segment and compare with the size of the segment.
        nb_1_GT = (segment_GT == 1).sum()
        nb_1_pred = (segment_pred == 1).sum()
        if nb_1_GT > threshold_pixels: # It means the segment is positive.
            if nb_1_pred > threshold_pixels: # It means the segment is predicted as positive.
                dict_counts['TP'] += 1
            else: 
                dict_counts['FN'] += 1
        else:
            if nb_1_pred > threshold_pixels:
                dict_counts['FP'] += 1
            else:
                dict_counts['TN'] += 1
            
    precision = dict_counts['TP'] / (dict_counts['TP'] + dict_counts['FP'])
    recall = dict_counts['TP'] / (dict_counts['TP'] + dict_counts['FN'])
                         
    ### Compute precision, recall
    return {'precision': precision, 'recall': recall}

def evaluate(ground_truth, predictions, evaluation_type, parameter_interval, list_image_names):
    # ground_truth should be a list of dictionary of the private elements with their list of polygons
    # priv_elem_GT = GT_annotation_to_polygon_dict(ground_truth)

    if evaluation_type == 'privacy_type':
        dict_iou = {}
        dict_pixel = {}
        for idx_im in range(len(predictions)):
            # Get the ground truth 
            # Name of the image
            im_name = list_image_names[idx_im]
            result_per_iou, result_per_pixel = evaluate_privacy(ground_truth[im_name], predictions[idx_im])
            for priv_elem in result_per_iou:
                if priv_elem not in dict_iou:
                    nb_pos = {'total_count': len(result_per_iou[priv_elem])}
                    for param_eval in parameter_interval:
                        if param_eval not in nb_pos:
                                nb_pos[param_eval] = 0
                        for result in result_per_iou[priv_elem]:
                            if result > param_eval:
                                nb_pos[param_eval] += 1 
                    
                    dict_iou[priv_elem] = nb_pos
                else:
                    dict_iou[priv_elem]['total_count'] += len(result_per_iou[priv_elem])
                    for param_eval in parameter_interval:
                        for result in result_per_iou[priv_elem]:
                            if result > param_eval:
                                dict_iou[priv_elem][param_eval] += 1 
                    
            for priv_elem in result_per_pixel:
                if priv_elem not in dict_pixel:
                    dict_pixel[priv_elem] = [result_per_pixel[priv_elem]]
                else:
                    dict_pixel[priv_elem].append(result_per_pixel[priv_elem])

                
        # And aggreagte for all the instances into a final score
        score_iou_list = {}
        for priv_elem in dict_iou:
            dict_per_threshold = {}
            for param_eval in dict_iou[priv_elem]:
                if param_eval != 'total_count':
                    dict_per_threshold[param_eval] = dict_iou[priv_elem][param_eval] / dict_iou[priv_elem][total_count]
            score_iou_list[priv_elem] = dict_per_threshold
        score_pixel_list = {}
        for priv_elem in dict_pixel:
            score_pixel_list[priv_elem] = np.mean(dict_pixel[priv_elem])
            
        return score_iou_list, score_pixel_list 
            
    elif evaluation_type == 'instance_type':
        dict_segment_result = {}
        for param_eval in parameter_interval:
            dict_segment_result[param_eval] = {'precision': [], 'recall': []}
            for idx_im in range(len(predictions)):
                im_name = list_image_names[idx_im]
                result = evaluate_instance(ground_truth[im_name], predictions[idx_im], param_eval)
                dict_segment_result[param_eval]['precision'].append(result['precision'])
                dict_segment_result[param_eval]['recall'].append(result['recall'])
            dict_segment_result[param_eval]['precision'] = np.mean(dict_segment_result[param_eval]['precision'])
            dict_segment_result[param_eval]['recall'] = np.mean(dict_segment_result[param_eval]['recall'])
        return dict_segment_result 