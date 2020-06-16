#def ruleBasedMapping(list_semantic_segmentation, list_OCR, list_scene):
import shapely
import json
import os
from shapely.geometry import mapping, shape
import matplotlib.pyplot as plt
from mxnet import image
from shapely.geometry import Polygon
from shapely.ops import unary_union

def post_process_polygons(list_polygons):
    # Merge overlapping polygons
    merge_poly = (unary_union(list_polygons))
    if type(merge_poly) == shapely.geometry.polygon.Polygon:
        return [merge_poly]
    else:
        return list(merge_poly)


def printPolygons(image_file, list_polygons, save_file=""):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    img = image.imread(image_file)
    ax.imshow(img.asnumpy()) #, extent=[0, 3000, 0, 3000])
    #ax.autoscale(False)
    for poly in list_polygons:
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y,  linewidth=5)
    #plt.show()
    if save_file != "":
        fig.savefig(save_file)
    plt.close(fig)
        
        
### We have to serialize the polygons with "mapping" to be able to save them. When we want to re-use them, we will need to transform them back with "shape".

# function to add to JSON 
def write_json(data, filename='data.json'): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 

def add_polygons_to_json(json_file, list_polygons, image_id):
    with open(json_file) as j_file: 
        data = json.load(j_file) 
        temp = data['list_image_poly'] 
        temp.append({image_id: list_polygons}) 
    write_json(data, json_file)   # Shouldn't it be temp???

def initialize_file(json_file):
    dict_image_polygons = {"list_image_poly": []}
    with open(json_file, 'w') as fp:
        json.dump(dict_image_polygons, fp)

def save_polygons(json_file, list_polygons, image_id):
    if not os.path.isfile(json_file):
        initialize_file(json_file)
    
    list_polygons_serialized = [mapping(polygons) for polygons in list_polygons]
    add_polygons_to_json(json_file, list_polygons_serialized, image_id)


    


# Check ground truth polygons

def prepare_GT_polygon(image_file, ground_truth_file):
    with open(Path(ground_truth_file), 'r') as f:
        ground_truth = json.load(f)
    ground_truth = ground_truth['annotations']
    image_ground_truth = ground_truth[Path(input_image_file).stem]["attributes"]
    # Make a list of the polygons
    list_polygons = []
    for element in image_ground_truth:
        for poly in element["polygons"]:
            list_polygons.append(eval_u.listCoordinates_to_shapelyPolygon(poly))
    return list_polygons