from pathlib import Path


import evaluation_utils as eval_u
import deeplab_utils as dl_u
import OCR_utils as OCR_u
import image_utils as image_u
import mapping_utils as mapping_u

import difflib
import os

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import mapping, shape
import shapely.wkt

import matplotlib.pyplot as plt

from mxnet import image

import json
import string
import numpy as np
import pandas as pd
import pickle
import random


import time

def writeResultsToJson(json_file, method_type, method_param, per_privacy_element_result,\
                       per_pixel_result):
    # If file does not exist.
    if not os.path.isfile(json_file):
        dict_results = {"list_result_methods": []}
        with open(json_file, 'w') as fp:
            json.dump(dict_results, fp)
    # Prepare what to append.
    results_to_add = {"method_type": method_type, "method_param": method_param,\
                     "privacy_element_result": per_privacy_element_result,\
                     "pixel_result": per_pixel_result}
    # Append.
    with open(json_file) as j_file: 
        data = json.load(j_file) 
        temp = data["list_result_methods"] 
        temp.append(results_to_add) 
    with open(json_file,'w') as f: 
        json.dump(data, f, indent=4)

from functools import singledispatch


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

def addElementToJson(element, json_file):
    if not os.path.isfile(json_file):
        with open(json_file, 'w') as fp:
            json.dump({"list_json": []}, fp)
    with open(json_file) as j_file: 
        data = json.load(j_file) 
        temp = data['list_json'] 
        temp.append({element[0]: element[1]}) 
    with open(json_file,'w') as f: 
        json.dump(data, f, indent=4, default=to_serializable)    

def getSceneJson(image_folder, list_image_files, output_json, dict_param):
    nb_im = 0
    for im in list_image_files:
        nb_im +=1
        print("Image no. ", str(nb_im), ": ", im)
        _, _, inputs_to_mapping_scene = \
                mapping_u.get_input_to_mapping(Path(image_folder + "/" + im), ["scene_recognition"], dict_param)
        addElementToJson((im, inputs_to_mapping_scene[0]), output_json)

#output_json = "200_training_scene_info.json"
#getSceneJson(image_folder, list_images, output_json, {"top_k":-1})



def getGroundTruthJson(image_folder, list_image_files, output_json, ground_truth_file):
    for im in list_image_files:
        addElementToJson((im, list(set(getGroundTruthLabel(im, ground_truth_file)))), output_json)

#getGroundTruthJson(image_folder, list_images, "GT_200_training_images.json", '../train2017.json')        


def getSemanticSegJson(image_folder, list_image_files, output_json):
    nb_im = 0
    for im in list_image_files:
        nb_im +=1
        print("Image no. ", str(nb_im), ": ", im)
        inputs_to_mapping_semantic_segmentation, _, _ = \
                mapping_u.get_input_to_mapping(Path(image_folder + "/" + im), ["semantic_segmentation"], [])
        addElementToJson((im, inputs_to_mapping_semantic_segmentation), output_json)
#getSemanticSegJson(image_folder, list_images, "200_training_semantic_seg_info.json")


def postProcessOCROutputs(OCR_outputs, needed_elements):
    OCR_processed_output = []
    for text_recognized in OCR_outputs:
                poly_text = text_recognized[0]
                #print(poly_text)
                possible_values = list(set(text_recognized[1]))
                #print(possible_values)
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
                            OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "hasNumbers"))
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
                            if ("LOCATION" in list_recognized_entities):
                                OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "LOCATION"))
                                break
                            elif ("PERSON" in list_recognized_entities):
                                OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "PERSON"))
                                break
                            elif ("ORGANIZATION" in list_recognized_entities):
                                OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "ORGANIZATION"))
                                break
                            elif ("GPE" in list_recognized_entities):
                                OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "GPE"))
                                break


                        # Obfuscate elements in a list of names or locations.
                        words = potential_value.split()
                        list_words = []
                        for value in words:
                            list_words += [value, value.upper(), value.lower(), value.title()]

                        for word in list_words:
                            similar_words = ""
                            if len(word) > 3: # This is a design choice to avoid small words...
                            # Get each word from the extracted strings and check for similarity
                                similar_words = difflib.get_close_matches(word, needed_elements["lines_names"], n=3, cutoff=0.9)
                                if len(similar_words) > 0:
                                    #print(potential_value, similar_words)
                                    OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "PERSON"))
                                    break
                                similar_words = difflib.get_close_matches(word, needed_elements["lines_location"], n=3, cutoff=0.9)
                                if len(similar_words) > 0:
                                    #print(potential_value, similar_words)
                                    OCR_processed_output.append((Polygon([ (poly_text[0], poly_text[1]), (poly_text[2], poly_text[1]),(poly_text[2], poly_text[3]),  (poly_text[0], poly_text[3])]), potential_value, "LOCATION"))
                                    break

                            if len(similar_words) > 0:
                                break
    return OCR_processed_output

def prepareOCRElements():
    dict_preparation = mapping_u.prepare_needed_elements(["OCR"])
    needed_input = mapping_u.prepare_OCR_name_matching()
    return dict_preparation, needed_input

def getOCRJson(image_folder, list_image_files, output_json, dict_preparation, needed_input, prepare_tools=True):        
    nb_im = 0
    for im in list_image_files:
        nb_im +=1
        print("Image no. ", str(nb_im), ": ", im)
        _, input_to_mapping_OCR, _ = \
                mapping_u.get_input_to_mapping(Path(image_folder + "/" + im), ["OCR"], dict_preparation)
        
        # POst process
        if len(input_to_mapping_OCR[0]) > 0:
            print("Post processing.")
            input_to_mapping_OCR_post_process = postProcessOCROutputs(input_to_mapping_OCR[0], needed_input)
        else:
            input_to_mapping_OCR_post_process = ()
        addElementToJson((im, input_to_mapping_OCR_post_process), output_json)
#prepare_elem = True
#if prepare_elem:
#    dict_preparation, needed_input = prepareOCRElements()
#getOCRJson(image_folder, list_images, "200_training_OCR_info.json", dict_preparation, needed_input, prepare_tools=True)




def openJsonInputs(file):
    with open(file) as f_in:
        data = json.load(f_in)
        data = data["list_json"]
        return data

### Create dataset


def createListsForDatasets(list_semantic, list_scene, list_OCR, list_GT_infos):
    all_element_GT = []
    all_element_semantic = []
    all_element_scene = []
    all_element_OCR = []

    dataset_semantic_unique = []
    dataset_scene_unique = []
    dataset_OCR_unique = []
    dataset_GT_unique = []


    dataset_semantic = []
    dataset_scene = []
    dataset_OCR = []
    dataset_GT = []

    for elem_semantic in (list_semantic):
        # Get row.
        row = []
        for elem in elem_semantic[list(elem_semantic.keys())[0]][0]:
            # If elements not already in the list, append them.
            if elem[1] not in all_element_semantic:
                all_element_semantic.append(elem[1])
            row.append(elem[1])  
        # Append row.
        dataset_semantic.append(row)
        dataset_semantic_unique.append(list(set(row)))

    for elem_scene in list_scene:
        row = elem_scene[list(elem_scene.keys())[0]][0][0][0]
        if row not in all_element_scene:
            all_element_scene.append(row)
        dataset_scene.append([row])
        dataset_scene_unique.append(list(set([row])))

    for elem_GT in list_GT_infos:
        row = elem_GT[list(elem_GT.keys())[0]]
        for elem_row in row:
            if elem_row not in all_element_GT:
                all_element_GT.append(elem_row)
        dataset_GT.append(row)
        dataset_GT_unique.append(list(set(row)))

    for elem_OCR in list_OCR:
        row = []
        for elem in elem_OCR[list(elem_OCR.keys())[0]]:
            if elem[2] not in all_element_OCR:
                all_element_OCR.append(elem[2])
            row.append(elem[2])
        dataset_OCR.append(row)
        dataset_OCR_unique.append(list(set(row)))
        
    return all_element_GT, all_element_semantic, all_element_scene, all_element_OCR, \
        dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, dataset_GT_unique, \
        dataset_semantic, dataset_scene, dataset_OCR, dataset_GT

def assembleDataLists(data_lists):
    # Create dataframes out of each list.
    list_df = [pd.DataFrame(l) for l in data_lists]
    # Concatenate the dataframes.
    df = pd.concat(list_df, axis=1, sort=False)
    # Get back a list.
    to_list = df.values.tolist()
    # Remove the nan values.
    clean_list = [list(filter(None.__ne__, l)) for l in to_list]
    return clean_list



def createDatasetAssociations(all_element_GT, all_element_semantic, all_element_scene, all_element_OCR, \
        dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, dataset_GT_unique, \
        dataset_semantic, dataset_scene, dataset_OCR, dataset_GT):
    
    dictionary_of_datasets = {}
    
    # All data unique.
    dictionary_of_datasets["semantic_scene_OCR_GT_unique"] = \
    {"data": assembleDataLists([dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, dataset_GT_unique]), \
     "not_in_antecedents": frozenset(all_element_GT),\
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene + all_element_OCR)}
    
    # Semantic, scene, GT unique.
    dictionary_of_datasets["semantic_scene_GT_unique"] = \
    {"data": assembleDataLists([dataset_semantic_unique, dataset_scene_unique, dataset_GT_unique]), \
     "not_in_antecedents": frozenset(all_element_GT),\
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene)}
    
    # Semantic, GT, unique.
    dictionary_of_datasets["semantic_GT_unique"] = \
    {"data": assembleDataLists([dataset_semantic_unique, dataset_GT_unique]), \
     "not_in_antecedents": frozenset(all_element_GT),\
     "not_in_consequents": frozenset(all_element_semantic)}
    
    # All data.
    dictionary_of_datasets["semantic_scene_OCR_GT"] = \
    {"data": assembleDataLists([dataset_semantic, dataset_scene, dataset_OCR, dataset_GT]), \
     "not_in_antecedents": frozenset(all_element_GT),\
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene + all_element_OCR)}
    
    # Semantic, scene, GT.
    dictionary_of_datasets["semantic_scene_GT"] = \
    {"data": assembleDataLists([dataset_semantic, dataset_scene, dataset_GT]), \
     "not_in_antecedents": frozenset(all_element_GT),\
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene)}
    
    # Semantic, GT.
    dictionary_of_datasets["semantic_GT"] = \
    {"data": assembleDataLists([dataset_semantic, dataset_GT]), \
     "not_in_antecedents": frozenset(all_element_GT),\
     "not_in_consequents": frozenset(all_element_semantic)}
    
    return dictionary_of_datasets



import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
def getRules(dataset, min_support_score=0.6, min_lift_score=1.2, min_confidence_score=0.75):
    # Get the frequent item set.
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support_score, use_colnames=True)

    # Post filter the rules, for instance to use two metrics
    #association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift_score)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[ (rules['antecedent_len'] >= 0) &
           (rules['confidence'] > min_confidence_score) &
           (rules['lift'] > min_lift_score) ]
    return rules, frequent_itemsets

def getPredictionRules(rules, list_not_in_consequents, list_not_in_antecedents):
    #rules[rules['consequents'] == {'Eggs', 'Kidney Beans'}]
    # Certain elements can not be in consequents:
    # The dataframes use frozensets!!!
    #lambda f: 1 if  len(f.intersection(['a','b']))>0 else 0
    idx = rules['consequents'].apply(lambda f: False if len(f.intersection(list_not_in_consequents))>0 else True)
    filtered_rules = rules.loc[idx, :]
    #filtered_rules = rules.loc(rules['consequents'].apply(lambda f: 0 ))#if len(f.intersection(list_not_in_consequents))>0 else 1))# 0 if  len(f["consequents"].intersection(list_not_in_consequents))>0 else 1)
    #filtered_rules = rules.loc[~rules['consequents'].isin(list_not_in_consequents)]
    # Certain elements can not be in antecedents:
    return filtered_rules.loc[filtered_rules['antecedents'].apply(lambda f: False if len(f.intersection(list_not_in_antecedents))>0 else True)]


def postProcessRules(df_rules):
    # We need to convert the frozensets into lists.
    df_rules["antecedents"] = df_rules["antecedents"].apply(list)
    df_rules["consequents"] = df_rules["consequents"].apply(list)
    return df_rules

def saveRules(rule_dataframe, list_not_in_consequents, file_name_csv, file_name_p):
    rule_dataframe.to_csv(file_name_csv, index=False)
    with open(file_name_p, 'wb') as fp:
        pickle.dump(list_not_in_consequents, fp)


### Create dataset


#P = shapely.wkt.loads('POLYGON ((51.0 3.0, 51.3 3.61, 51.3 3.0, 51.0 3.0))')

def createListsForDatasetsEvaluation(list_semantic, list_scene, list_OCR, scene_top_k=1):
    
    all_element_semantic = []
    all_element_scene = []
    all_element_OCR = []

    #dataset_semantic_unique = []
    #dataset_scene_unique = []
    #dataset_OCR_unique = []
   
    dataset_semantic = []
    dataset_scene = []
    dataset_OCR = []
    
    for elem_semantic in (list_semantic):
        
        # Get row.
        row = []
        for elem in elem_semantic[list(elem_semantic.keys())[0]][0]:
            list_polygon = [poly[0] for poly in elem[0]]
            
            list_to_append = [[shapely.wkt.loads(poly), elem[1]] for poly in list_polygon]
            # If elements not already in the list, append them.
            if elem[1] not in all_element_semantic:
                all_element_semantic.append(elem[1])
            row += list_to_append
        # Append row.
        dataset_semantic.append(row)
        #dataset_semantic_unique.append(list(set(row)))

    for elem_scene in list_scene:
        #print(elem_scene[list(elem_scene.keys())[0]][0])
        row = [[None, scene_name[0]] for scene_name in elem_scene[list(elem_scene.keys())[0]][0][0:scene_top_k]]#elem_scene[list(elem_scene.keys())[0]][0][0][0]
        list_scene_names =  [scene_name[0] for scene_name in elem_scene[list(elem_scene.keys())[0]][0][0:scene_top_k]]#elem_scene[list(elem_scene.keys())[0]][0][0][0]
        for scene_name in list_scene_names:
            if scene_name not in all_element_scene:
                all_element_scene.append(scene_name)
        dataset_scene.append(row)
        #dataset_scene_unique.append(list(set([row])))


    for elem_OCR in list_OCR:
        row = []
        for elem in elem_OCR[list(elem_OCR.keys())[0]]:
            #print(elem[0])
            if elem[2] not in all_element_OCR:
                all_element_OCR.append(elem[2])
            row.append([shapely.wkt.loads(elem[0]), elem[2]])
        dataset_OCR.append(row)
        #dataset_OCR_unique.append(list(set(row)))
        
    return all_element_semantic, all_element_scene, all_element_OCR, \
        dataset_semantic, dataset_scene, dataset_OCR
        #dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique,  \
        



def assembleDataListswithPolygons(data_lists):
    # Create dataframes out of each list.
    list_df = [pd.DataFrame(l) for l in data_lists]
    # Concatenate the dataframes.
    df = pd.concat(list_df, axis=1, sort=False)
    # Get back a list.
    to_list = df.values.tolist()
    # Remove the nan values.
    clean_list = [list(filter(None.__ne__, l)) for l in to_list]
    return clean_list

def createDatasetAssociationsForEvaluation(all_element_semantic, all_element_scene, all_element_OCR, \
        dataset_semantic, dataset_scene, dataset_OCR): #dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, \
    
    dictionary_of_datasets = {}
    
    # All data unique.
    """
    dictionary_of_datasets["semantic_scene_OCR_unique"] = \
    {"data": assembleDataLists([dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique]), \
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene + all_element_OCR)}
    
    # Semantic, scene unique.
    dictionary_of_datasets["semantic_scene_unique"] = \
    {"data": assembleDataLists([dataset_semantic_unique, dataset_scene_unique]), \
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene)}
    
    # Semantic unique.
    dictionary_of_datasets["semantic_unique"] = \
    {"data": assembleDataLists([dataset_semantic_unique]), \
     "not_in_consequents": frozenset(all_element_semantic)}
    """
    # All data.
    dictionary_of_datasets["semantic_scene_OCR"] = \
    {"data": assembleDataListswithPolygons([dataset_semantic, dataset_scene, dataset_OCR]), \
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene + all_element_OCR)}
    
    # Semantic, scene.
    dictionary_of_datasets["semantic_scene"] = \
    {"data": assembleDataListswithPolygons([dataset_semantic, dataset_scene]), \
     "not_in_consequents": frozenset(all_element_semantic + all_element_scene)}
    
    # Semantic.
    dictionary_of_datasets["semantic"] = \
    {"data": assembleDataListswithPolygons([dataset_semantic]), \
     "not_in_consequents": frozenset(all_element_semantic)}
    
    return dictionary_of_datasets



# Extract the rule function.

def getListsIntersections(list1, list2):
    #print("TODO: we only handle entries with a single elements of each class.")
    list1 = list(set(list1))
    list2 = list(set(list2))
    list_intersection = []
    list_leftovers_list2 = list2.copy()
    list_leftovers_list1 = list1.copy()
    for elem1 in list1:
        if elem1 in list2:
            list_intersection.append(elem1)
            list_leftovers_list2.remove(elem1)
            list_leftovers_list1.remove(elem1)
    return len(list_intersection), len(list_leftovers_list1), len(list_leftovers_list2)
 
def getScoreFromLists(nb_intersect, nb_leftover_1, nb_leftover_2, nb_max_elem):
    # We could also add weights but I think for now this is not needed.
    return nb_intersect + (nb_max_elem - nb_leftover_1) + (nb_max_elem - nb_leftover_2)

def getScore(list1, list2, nb_max_elem):
    nb_intersect, nb_leftover_1, nb_leftover_2 = getListsIntersections(list1, list2)
    score = getScoreFromLists(nb_intersect, nb_leftover_1, nb_leftover_2, nb_max_elem)
    return score
    
def dataFrameRulesToMapping(dataframe, list_not_in_consequents):
    print("TODO: we only handle entries with a single elements of each class.")
    def mappingFunction(inputs_list, threshold):
        # Inputs_list should be a list of polygons and the corresponding labels.
        #inputs_list = inputs_list[0]
        #inputs_list_labels = [item[1] for item in inputs_list]
        #print(inputs_list_labels)
        df_for_selection = dataframe.copy(deep=True)
        # Compute intersection.
        # Compute number of leftover rules.
        # Compute score for each rule.
        list_polygons_to_obfuscate = []
        for input_image in inputs_list:
            list_labels = [item[1] for item in input_image]
            df_for_selection["score_rule"] = df_for_selection["antecedents"].apply(lambda x: getScore(list_labels, x, len(list_not_in_consequents))) 
            # Select list of rules that are applicable.
            ################## This should be row by row!!!!!!!!!!!!!!!
            list_antecedents = df_for_selection[df_for_selection["score_rule"] >= threshold]["antecedents"].tolist()
            list_antecedents_flat = list(set([item for sublist in list_antecedents for item in sublist]))
            
            # Get the corresponding polygons.
            
            #for elem in inputs_list:
            #    if elem[1] in list_antecedents_flat:
            #        list_polygons_to_obfuscate.append(elem[0])
            poly_to_obfuscate_per_image = []
            for poly, poly_label in input_image:
                if poly_label in list_antecedents_flat:
                    poly_to_obfuscate_per_image.append(poly)
            list_polygons_to_obfuscate.append(poly_to_obfuscate_per_image)
        # Return polygons to obfuscate.
        return list_polygons_to_obfuscate
        
    return mappingFunction





def getTrainingDataSegment(nb_data, list_semantic, list_scene, list_GT_infos, list_OCR):
    # Get random indices.
    sampled_indices = random.sample(range(len(list_scene)), nb_data)
    # Get the new lists.
    new_list_semantic = [list_semantic[i] for i in sampled_indices] 
    new_list_scene = [list_scene[i] for i in sampled_indices] 
    new_list_OCR = [list_OCR[i] for i in sampled_indices] 
    new_list_GT_infos = [list_GT_infos[i] for i in sampled_indices] 
    
    return new_list_semantic, new_list_scene, new_list_GT_infos, new_list_OCR

def preProcessTrainingData(list_semantic, list_scene, list_GT_infos, list_OCR):
    all_element_GT, all_element_semantic, all_element_scene, all_element_OCR, \
            dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, dataset_GT_unique, \
            dataset_semantic, dataset_scene, dataset_OCR, dataset_GT = \
            createListsForDatasets(list_semantic, list_scene, list_OCR, list_GT_infos)

    dictionary_of_datasets = createDatasetAssociations(all_element_GT, all_element_semantic, all_element_scene, all_element_OCR, \
            dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, dataset_GT_unique, \
            dataset_semantic, dataset_scene, dataset_OCR, dataset_GT)
    
    return dictionary_of_datasets








def main():
    ## Training data.
    list_semantic_file = "200_training_semantic_seg_info.json"
    list_GT_infos_file = "GT_200_training_images.json"
    list_scene_file = "200_training_scene_info.json"
    list_OCR_file = "200_training_OCR_info.json"

    list_semantic = openJsonInputs(list_semantic_file)
    list_scene = openJsonInputs(list_scene_file)
    list_GT_infos = openJsonInputs(list_GT_infos_file)
    list_OCR = openJsonInputs(list_OCR_file)

    # Read the ground truth file.
    with open(Path('../val2017.json'), 'r') as f:
        ground_truth = json.load(f)
    ground_truth = ground_truth['annotations']
    segment_list= [2, 4, 8, 12, 16, 20]


    ### Test data.
    # Get the evaluation data with their encoding for rules.
    list_semantic_file_eval = "eval_semantic_seg_info.json"
    list_scene_file_eval = "eval_scene_info.json"
    list_OCR_file_eval = "eval_OCR_info.json"
    list_semantic_eval = openJsonInputs(list_semantic_file_eval)
    list_scene_eval = openJsonInputs(list_scene_file_eval)
    list_OCR_eval = openJsonInputs(list_OCR_file_eval)

    all_element_semantic, all_element_scene, all_element_OCR, \
            dataset_semantic, dataset_scene, dataset_OCR = \
            createListsForDatasetsEvaluation(list_semantic_eval, list_scene_eval, list_OCR_eval)
    eval_dictionary_of_datasets = createDatasetAssociationsForEvaluation(all_element_semantic, all_element_scene, all_element_OCR, \
            dataset_semantic, dataset_scene, dataset_OCR) # dataset_semantic_unique, dataset_scene_unique, dataset_OCR_unique, \

    with open("list_eval_images.json") as json_file:
        list_images = json.load(json_file)




    ### Parameters for evaluation.
    nb_training_data = [100, 150, 200]
    rule_data_type = ["semantic_scene_GT_unique", "semantic_scene_OCR_GT_unique", "semantic_GT_unique"]
    rules_support = [0.1, 0.4, 0.7] # done for 0.7 nothing works, 0.7]
    rules_lift = [0.0, 0.3, 0.6, 0.9, 1.1]
    rules_confidence = [0.3, 0.6, 0.9]#[0.0, 0.3, 0.6, 0.9]
    threshold_rules_to_keep = [0.0, 0.3, 0.6]

    ### Evaluation loop.
    folder_saving = "./results/"
    if not os.path.exists(folder_saving):
        os.makedirs(folder_saving)
        
    # Result file.
    json_file_results = Path(folder_saving+ "results_exp_automatic.json")

    for data_segment_nb in nb_training_data:
        # Get training data.
        new_list_semantic, new_list_scene, new_list_GT_infos, new_list_OCR = getTrainingDataSegment(data_segment_nb, list_semantic, list_scene, list_GT_infos, list_OCR)
        dictionary_of_datasets = preProcessTrainingData(new_list_semantic, new_list_scene, new_list_GT_infos, new_list_OCR)
        # Get rules.
        for data_names in rule_data_type:
            for param_rules_support in rules_support:
                for param_rules_lift in rules_lift:
                    for param_rules_confidence in rules_confidence:
                        rules, frequent_itemsets = getRules(dictionary_of_datasets[data_names]["data"], param_rules_support, param_rules_lift, param_rules_confidence)
                        filtered_rules = getPredictionRules(rules, dictionary_of_datasets[data_names]["not_in_consequents"], dictionary_of_datasets[data_names]["not_in_antecedents"])
                        d = postProcessRules(filtered_rules.copy())
                        if len(d) > 1:
                            rules_func = dataFrameRulesToMapping(d, dictionary_of_datasets[data_names]["not_in_consequents"])
                            # Apply the rules to the test data.
                            data_names_test = data_names
                            data_names_test = data_names_test.replace('_GT_unique','')
                            for rule_threshold in threshold_rules_to_keep:
                                method_type = "mined_rules"
                                method_param = {"nb_training_data": data_segment_nb, "type_data": data_names,\
                                               "support_thresh":param_rules_support,\
                                               "lift_thresh":param_rules_lift,\
                                               "confidence_thresh":param_rules_confidence,\
                                               "rule_thresh":rule_threshold} 
                                print("Start dealing with ", method_param)
                                list_poly_to_obfuscate = rules_func(eval_dictionary_of_datasets[data_names_test]["data"], rule_threshold)
                                # Save the polygons.                
                                image_folder = "../dataset_images"
                                file_path_saving = "mined_" + str(data_segment_nb) + "_" +\
                                data_names + "_" + str(param_rules_support) + "_" + str(param_rules_lift) +\
                                "_" + str(param_rules_confidence) + "_" + str(rule_threshold)
                                if not os.path.exists(Path(folder_saving + file_path_saving)):
                                    os.makedirs(Path(folder_saving+file_path_saving))
                                print(os.path.isdir(Path(folder_saving+file_path_saving)))
                                print(os.path.abspath(Path(folder_saving+file_path_saving)))
                                
                                for image_name, polys in zip(list_images, list_poly_to_obfuscate):
                                    #print(image_name)
                                    polys = list(filter(lambda a: a != None, polys))
                                    polys_clean = [] 
                                    for poly in polys:
                                        if poly.type == "Polygon":
                                            polys_clean.append(poly)
                                    polys = polys_clean 
                                    if len(polys) > 1:
                                        polys_to_obfuscate_final = image_u.post_process_polygons(polys)
                                    else:
                                        polys_to_obfuscate_final = polys
                                    # Remove empty polygons.
                                    polys_clean = [] 
                                    for poly in polys_to_obfuscate_final:
                                        if not poly.is_empty:
                                            polys_clean.append(poly)
                                    polys_to_obfuscate_final = polys_clean
                                    while True:
                                        n = 0
                                        break_ = False
                                        while True:
                                            n = n+1
                                            try:  
                                                image_u.save_polygons(Path(folder_saving+file_path_saving + ".json"), polys_to_obfuscate_final, Path(image_name).stem )
                                                
                                            except:
                                                print("looping ", n)
                                                if (n>50):
                                                    break_ = True
                                                    break
                                                pass
                                            else:
                                                break_ = True
                                                break
                                            
                                        if (n>50) or break_:
                                            break                                # Save the images.
                                    save_file = Path(folder_saving + file_path_saving + "/") / (Path(image_name).stem + ".png")
                                    input_image_file = Path(image_folder + "/" + image_name)
                                    #printPolygons(input_image_file, polys_to_obfuscate, save_file)
                                    image_u.printPolygons(input_image_file, polys_to_obfuscate_final, save_file)
                                del list_poly_to_obfuscate, polys_to_obfuscate_final, input_image_file
                                del poly, polys, polys_clean
                                # Compute evaluation measures.
                                with open(Path(folder_saving + file_path_saving + ".json"), 'r') as f:
                                    list_predictions = json.load(f)
                                list_predictions = list_predictions["list_image_poly"]
                                prediction_list = list_predictions
                                ground_truth_list = ground_truth
                                print("Computing the results.")
                                rate_obuscation_per_privacy_element = eval_u.evaluationPerPrivacyElement(prediction_list, ground_truth_list)
                                dict_results = eval_u.evaluationPerPixel(prediction_list, ground_truth_list, segment_list)
                                dict_results_agg = eval_u.aggregateResultsPrivacyElement(dict_results)
                                dict_results_agg_no_nan = eval_u.aggregateResultsPrivacyElement(dict_results, "nonan")
                                # Save evaluation measures.   
                                del list_predictions, prediction_list, ground_truth_list
                                
                                per_privacy_element_result = [dict_results_agg, dict_results_agg_no_nan]
                                per_pixel_result = dict_results
                                writeResultsToJson(json_file_results, method_type, method_param, per_privacy_element_result,\
                                                       per_pixel_result)
                                print("Finished dealing with ", method_param)
                                del per_privacy_element_result
                                del per_pixel_result, dict_results_agg, dict_results_agg_no_nan, dict_results, rate_obuscation_per_privacy_element
                                
                        else:
                            method_type = "mined_rules"
                            method_param = {"nb_training_data": data_segment_nb, "type_data": data_names,\
                                               "support_thresh":param_rules_support,\
                                               "lift_thresh":param_rules_lift,\
                                               "confidence_thresh":param_rules_confidence,\
                                               "rule_thresh":"NA"} 
                            print(method_param)
                            per_privacy_element_result = ["NA", "NA"]
                            per_pixel_result = "NA"
                            writeResultsToJson(json_file_results, method_type, method_param, per_privacy_element_result,\
                                                       per_pixel_result)

if __name__ == '__main__':
	main()