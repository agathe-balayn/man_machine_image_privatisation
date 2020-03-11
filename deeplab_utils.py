import numpy as np
from PIL import Image
import cv2 as cv
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes as imfill
import skimage.transform as sk_t



def find_class_name(x):
    # ADE dataset classes # numbered from 1!
    CLASSES = ("wall", "building, edifice", "sky", "floor, flooring", "tree",
               "ceiling", "road, route", "bed", "windowpane, window", "grass",
               "cabinet", "sidewalk, pavement",
               "person, individual, someone, somebody, mortal, soul",
               "earth, ground", "door, double door", "table", "mountain, mount",
               "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
               "chair", "car, auto, automobile, machine, motorcar",
               "water", "painting, picture", "sofa, couch, lounge", "shelf",
               "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
               "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
               "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
               "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
               "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
               "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
               "grandstand, covered stand", "path", "stairs, steps", "runway",
               "case, display case, showcase, vitrine",
               "pool table, billiard table, snooker table", "pillow",
               "screen door, screen", "stairway, staircase", "river", "bridge, span",
               "bookcase", "blind, screen", "coffee table, cocktail table",
               "toilet, can, commode, crapper, pot, potty, stool, throne",
               "flower", "book", "hill", "bench", "countertop",
               "stove, kitchen stove, range, kitchen range, cooking stove",
               "palm, palm tree", "kitchen island",
               "computer, computing machine, computing device, data processor, "
               "electronic computer, information processing system",
               "swivel chair", "boat", "bar", "arcade machine",
               "hovel, hut, hutch, shack, shanty",
               "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
               "motorcoach, omnibus, passenger vehicle",
               "towel", "light, light source", "truck, motortruck", "tower",
               "chandelier, pendant, pendent", "awning, sunshade, sunblind",
               "streetlight, street lamp", "booth, cubicle, stall, kiosk",
               "television receiver, television, television set, tv, tv set, idiot "
               "box, boob tube, telly, goggle box",
               "airplane, aeroplane, plane", "dirt track",
               "apparel, wearing apparel, dress, clothes",
               "pole", "land, ground, soil",
               "bannister, banister, balustrade, balusters, handrail",
               "escalator, moving staircase, moving stairway",
               "ottoman, pouf, pouffe, puff, hassock",
               "bottle", "buffet, counter, sideboard",
               "poster, posting, placard, notice, bill, card",
               "stage", "van", "ship", "fountain",
               "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
               "canopy", "washer, automatic washer, washing machine",
               "plaything, toy", "swimming pool, swimming bath, natatorium",
               "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
               "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
               "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
               "trade name, brand name, brand, marque", "microwave, microwave oven",
               "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
               "bicycle, bike, wheel, cycle", "lake",
               "dishwasher, dish washer, dishwashing machine",
               "screen, silver screen, projection screen",
               "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
               "traffic light, traffic signal, stoplight", "tray",
               "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
               "dustbin, trash barrel, trash bin",
               "fan", "pier, wharf, wharfage, dock", "crt screen",
               "plate", "monitor, monitoring device", "bulletin board, notice board",
               "shower", "radiator", "glass, drinking glass", "clock", "flag")
    return CLASSES[x]


def create_mask_segments(prediction):
    # Get unique possible values
    unique_val = np.unique(prediction)
    #list_masks_dec = []
    list_masks_rgb = []
    list_class = []
    list_idx_class = []
    print("Number of classes with masks to create: ", unique_val.shape[0])
    for x in np.nditer(unique_val):
        #print("Creating mask for class ", x)
        list_idx_class.append(int(x))
        list_class.append(find_class_name(int(x)))
        #new_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3))
        # Where we find the value, we set it to 1 to become white.
        #idx_mask = np.where(prediction == x)
        #masked_im_dec = (np.where(prediction == x, 0.99, 0))
        masked_im_rgb = (np.where(prediction == x, 255, 0))
        #print(masked_im)
        #new_mask_dec = np.repeat(masked_im_dec[:, :, np.newaxis], 3, axis=2) # np.expand_dims(masked_im, axis=2)
        new_mask_rgb = np.repeat(masked_im_rgb[:, :, np.newaxis], 3, axis=2)
        #print(new_mask)
        #list_masks_dec.append(new_mask_dec)
        list_masks_rgb.append(new_mask_rgb)
    #return list_masks_dec, list_masks_rgb, list_class
    return list_masks_rgb, list_class, list_idx_class

def contour_to_polygon(output_contours):
    list_polygon = []
    for contour in output_contours:
        contour = np.squeeze(contour)
        if len(contour) > 2:
            list_polygon.append(Polygon(contour))
    return list_polygon

def get_polygon_confidence(poly, proba_matrix, class_idx):
    poly_coordinates_x, poly_coordinates_y = poly.exterior.coords.xy
    poly_coordinates_x = np.frombuffer(poly_coordinates_x).astype(int)
    poly_coordinates_y = np.frombuffer(poly_coordinates_y).astype(int)
    mask = np.zeros((proba_matrix.shape[2], proba_matrix.shape[3]), dtype=bool)
    mask[poly_coordinates_y, poly_coordinates_x] = 1
    img_proba = proba_matrix[0, class_idx, :, :]
    total_proba = img_proba[imfill(mask)].sum() / img_proba[imfill(mask)].shape[0]
    return total_proba


def deeplab_pred_to_output(prediction, _plot=False, compute_proba=False, proba_matrix=[] , size_back=False, list_shapes=0):
    # It actually taakes far too long to resize first  the proba matrix for the polygons....
    ### Check maybe i can just change the polygons coordinates.


    #print("TODO: manage resizing of images. -> this is done!")
    print("Creating masks.")
    if size_back:
      # REsize back to original size.
      print("Resizing image prediction.")
      prediction_original = prediction.asnumpy()
      prediction = sk_t.resize(prediction_original, (list_shapes[0], list_shapes[1]), anti_aliasing=False,order=0)
      new_im_rgb, list_class, list_idx_class = (create_mask_segments(prediction))


      if compute_proba:
          new_im_rgb_original, list_class_original, list_idx_class_original = (create_mask_segments(prediction_original))
          for mask, class_name, class_idx in zip(new_im_rgb_original, list_class_original, list_idx_class_original):
              #print("--getting contour")
              new_im1 = Image.fromarray(mask.astype('uint8'))
              new_im1_cv = cv.cvtColor(np.array(new_im1), cv.COLOR_RGB2BGR)
              imgray = cv.cvtColor(new_im1_cv, cv.COLOR_BGR2GRAY)
              ret, thresh = cv.threshold(imgray, 127, 255, 0)
              contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

              if _plot:
                  plt.imshow(thresh)
                  cv.drawContours(new_im1_cv, contours, -1, (0,255,0), 3)
                  plt.imshow( new_im1_cv)
                  plt.show()
              #print("--transforming contour to poygon.")

              list_shapely_polygon_original = contour_to_polygon(contours)

      #if compute_proba:
      #  proba_matrix = sk_t.resize(proba_matrix, (proba_matrix.shape[0], proba_matrix.shape[1], list_shapes[0], list_shapes[1]))
    else:
      new_im_rgb, list_class, list_idx_class = (create_mask_segments(prediction.asnumpy()))
    # new_im_dec, new_im_rgb, list_class = (create_mask_segments(prediction))
    #print("Creating polygons.")
    list_polygons = []
    for mask, class_name, class_idx in zip(new_im_rgb, list_class, list_idx_class):
        #print("--getting contour")
        new_im1 = Image.fromarray(mask.astype('uint8'))
        new_im1_cv = cv.cvtColor(np.array(new_im1), cv.COLOR_RGB2BGR)
        imgray = cv.cvtColor(new_im1_cv, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if _plot:
            plt.imshow(thresh)
            cv.drawContours(new_im1_cv, contours, -1, (0,255,0), 3)
            plt.imshow( new_im1_cv)
            plt.show()
        #print("--transforming contour to poygon.")

        list_shapely_polygon = contour_to_polygon(contours)
        #print("computing proba")
        if compute_proba:
            # Go through the polygon and get the average probability per polygon
            list_poly_proba = []

            if size_back:
                # Compute non-resized polygons as well
                for poly_original, poly_correct_size in zip(list_shapely_polygon_original, list_shapely_polygon):
                    total_proba = get_polygon_confidence(poly_original, proba_matrix, class_idx)
                    list_poly_proba.append((poly_correct_size, total_proba))
            else:
                #print("TODO: compute proba per polygon (per pixel?)")
                for poly in list_shapely_polygon:
                    #poly_coordinates_x, poly_coordinates_y = poly.exterior.coords.xy
                    #poly_coordinates_x = np.frombuffer(poly_coordinates_x).astype(int)
                    #poly_coordinates_y = np.frombuffer(poly_coordinates_y).astype(int)
                    #mask = np.zeros((prediction.shape[0], prediction.shape[1]), dtype=bool)
                    #mask[poly_coordinates_y, poly_coordinates_x] = 1
                    #img_proba = proba_matrix[0, class_idx, :, :]
                    #total_proba = img_proba[imfill(mask)].sum() / img_proba[imfill(mask)].shape[0]

                    

                    total_proba = get_polygon_confidence(poly, proba_matrix, class_idx)
                    list_poly_proba.append((poly, total_proba))
            list_polygons.append((list_poly_proba, class_name, class_idx))
        else:
            list_polygons.append((list_shapely_polygon, class_name, class_idx))
    
    return list_polygons