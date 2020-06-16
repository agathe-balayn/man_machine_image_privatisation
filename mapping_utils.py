from pathlib import Path

import evaluation_utils as eval_u
import deeplab_utils as dl_u
import OCR_utils as OCR_u
import image_utils as image_u
import difflib
import os

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import mapping, shape


from mxnet import image

import json
import string
import numpy as np

import time


def get_input_to_mapping(input_image_file, methods_to_account, dict_preparation):
    inputs_to_mapping_semantic_segmentation = []
    inputs_to_mapping_OCR = []
    inputs_to_mapping_scene = []

    for automatic_method in methods_to_account:
        if automatic_method == "semantic_segmentation":
            # Get the outputs of the semantic segmentation.
            print("TODO: choose whether to get the probability")
            list_img = []
            list_img_name = []
            list_img_ratio = []
            list_img_shapes = []
            list_output = []
            img, img_name, img_ratio, img_shapes = eval_u.prepare_sample(Path(input_image_file), 'deeplab', True)
            # Get predictions
            model_deeplab = eval_u.load_model('deeplab')
            output_pred_deeplab = eval_u.get_predictions([img], 'deeplab', model_deeplab)
            # Post process to polygon
            #output = dl_u.deeplab_pred_to_output(output_pred_deeplab[0][1], False, True, output_pred_deeplab[0][0])
            inputs = dl_u.deeplab_pred_to_output(output_pred_deeplab[0][1], False, True, output_pred_deeplab[0][0], True, img_shapes)
            list_img.append(img)
            list_img_name.append(img_name)
            list_img_ratio.append(img_ratio)
            list_img_shapes.append(img_shapes)
            list_output.append(inputs)
            inputs_to_mapping_semantic_segmentation.append(inputs)

        elif automatic_method == "OCR":
            # GEt the outputs of the Optical Character Recognition.
            output_pred_OCR = eval_u.get_predictions([str(input_image_file)], 'OCR', True, False)
            # Process for misspellings
            inputs = OCR_u.accountForMisspellings(output_pred_OCR[0], dict_preparation["words_dict"], dict_preparation["ss"])
            inputs_to_mapping_OCR.append(inputs)

        elif automatic_method == "scene_recognition":
            output_pred = eval_u.get_predictions([str(input_image_file)], 'vgg_places365', True, False, dict_preparation) #["top_k"])
            inputs_to_mapping_scene.append(output_pred)
            
    return inputs_to_mapping_semantic_segmentation, inputs_to_mapping_OCR, inputs_to_mapping_scene


    
def prepare_needed_elements(methods_to_account):
    if "OCR" in methods_to_account:
        ss, words_dict = OCR_u.prepareDictForMisspellings()
    return {"ss": ss, "words_dict": words_dict}




def prepare_OCR_name_matching():
    # Obfuscate any elemt in a list of location and persons.
    # load list of first names
    with open('names_countries/fnames.txt', encoding="utf8") as f:
        lines = f.read().splitlines()
    #load list of last names
    with open('names_countries/lnames.txt', encoding="utf8") as f:
        lines += f.read().splitlines()
    # load list of locations
    with open('names_countries/cities_and_countries.txt', encoding="utf8") as f:
        lines_place = f.read().splitlines()
    total_lines = lines + lines_place
    return {"lines_names": lines, "lines_location": lines_place, "lines": total_lines}



def ruleBasedMapping(type_semantic_seg_rule, type_OCR_rule, type_scene_rule, list_semantic_segmentation, list_OCR, list_scene, needed_elements):
    polys_to_obfuscate = []

    if len(list_semantic_segmentation) > 0:
        if type_semantic_seg_rule == "simple_list":
            print("Dealing with the polygons from semantic segmentation.")
            list_private_deeplab_labels = ["person, individual, someone, somebody, mortal, soul", \
                                       "car, auto, automobile, machine, motorcar", \
                       "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, ", \
                       "motorcoach, omnibus, passenger vehicle", "truck, motortruck", "van", 
                       "conveyer belt, conveyor belt, conveyer, conveyor, transporter",  "minibike, motorbike", \
                       "bicycle, bike, wheel, cycle", "poster, posting, placard, notice, bill, card", \
                       "signboard sign", "bulletin board, notice board", \
                      "screen door, screen",  "screen, silver screen, projection screen", \
                      "crt screen", "plate", "monitor, monitoring device", \
                       "bookcase", "blind, screen", "book", "computer, computing machine, computing device, data processor ", \
                        "electronic computer, information processing system", \
                        "television receiver, television, television set, tv, tv set, idiot ", \
                        "trade name, brand name, brand, marque", "flag"]
            for poly in list_semantic_segmentation:
                #print("TODO: add filter per confidence score")
                if poly[1] in list_private_deeplab_labels:
                    for poly_elem in poly[0]:
                        if poly_elem[0].area > 4.0: # Check that the size of the polygons is large enough to actually see anything on the images.
                            #print(poly[1])
                            polys_to_obfuscate.append(poly_elem[0])
                            #print("TODO: check a surface size to filter out polygons.")

    if len(list_OCR) > 0:
        if type_OCR_rule == "simple_rule":
            print("Dealing with the polygons from OCR.")

            for text_recognized in list_OCR:
                poly_text = text_recognized[0]
                #print(poly_text)
                possible_values = text_recognized[1]
                for potential_value in possible_values:
                    # Check whether the string is actually not just one letter or a space.
                    string_without_space = potential_value.translate({ord(c): None for c in string.whitespace})
                    if (len(string_without_space) > 1):
                        #print(potential_value)

                        ### Obfuscate any number
                        # count number of digits in the string:
                        nb_digit = sum(list(map(lambda x:1 if x.isdigit() else 0,set(potential_value))))
                        if nb_digit > 3 : # This is a parameter to tune. for now, 4 corresponds to a year, we will put 6 digits minimum because it corresponds to a birth date and phone numbers have even more numbers.
                            #print(potential_value)
                            polys_to_obfuscate.append(Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]))
                            break

                        # Obfuscate any element recognized as a location or organization or person.
                        continuous_chunk_1 = OCR_u.NERWithOldStanford(potential_value)
                        continuous_chunk_2 = OCR_u.NERNewVersion(potential_value)
                        list_recognized_entities_1 = [chunk[1] for chunk in continuous_chunk_1]
                        list_recognized_entities_2 = [chunk[1] for chunk in continuous_chunk_2]
                        list_recognized_entities = list_recognized_entities_1 + list_recognized_entities_2
                        if ("LOCATION" in list_recognized_entities) or \
                        ("PERSON" in list_recognized_entities) or \
                        ("ORGANIZATION" in list_recognized_entities) or \
                        ("GPE" in list_recognized_entities) :
                            #print(potential_value, list_recognized_entities)
                            polys_to_obfuscate.append(Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]))
                            break


                        # Obfuscate elements in a list of names or locations.
                        words = potential_value.split()
                        list_words = []
                        for value in words:
                            list_words += [value, value.upper(), value.lower(), value.title()]

                        for word in list_words:
                            # Get each word from the extracted strings and check for similarity
                            similar_words = difflib.get_close_matches(word, needed_elements["lines"], n=3, cutoff=0.9)
                            if len(similar_words) > 0:
                                #print(potential_value, similar_words)
                                polys_to_obfuscate.append(Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]))
                                break

                        if len(similar_words) > 0:
                            break

                        # Obfuscate elements that are next to "name" or "date". # Let's thnk about that later...

                        #print("TO IMPLEMENT")

        elif type_OCR_rule == "simplest_rule":
            print("Dealing with the polygons from OCR.")

            # Obfuscate all text that is diffeernt from ""
            for text_recognized in list_OCR:
                poly_text = text_recognized[0]
                possible_values = text_recognized[1]
                for potential_value in possible_values:
                    if potential_value.strip():
                        polys_to_obfuscate.append(Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]))
                        break

    return polys_to_obfuscate
