import numpy as np
from PIL import Image
import cv2 as cv
from shapely.geometry import Polygon
from matplotlib import pyplot as plt



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
    list_masks_dec = []
    list_masks_rgb = []
    list_class = []

    for x in np.nditer(unique_val):
        list_class.append(find_class_name(int(x)))
        #new_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3))
        # Where we find the value, we set it to 1 to become white.
        #idx_mask = np.where(prediction == x)
        masked_im_dec = (np.where(prediction == x, 0.99, 0))
        masked_im_rgb = (np.where(prediction == x, 255, 0))
        #print(masked_im)
        new_mask_dec = np.repeat(masked_im_dec[:, :, np.newaxis], 3, axis=2) # np.expand_dims(masked_im, axis=2)
        new_mask_rgb = np.repeat(masked_im_rgb[:, :, np.newaxis], 3, axis=2)
        #print(new_mask)
        list_masks_dec.append(new_mask_dec)
        list_masks_rgb.append(new_mask_rgb)
    return list_masks_dec, list_masks_rgb, list_class


def contour_to_polygon(output_contours):
    list_polygon = []
    for contour in output_contours:
        contour = np.squeeze(contour)
        if len(contour) > 2:
            list_polygon.append(Polygon(contour))
    return list_polygon


def deeplab_pred_to_output(prediction, _plot=False):
    new_im_dec, new_im_rgb, list_class = (create_mask_segments(prediction))
    list_polygons = []
    for mask, class_name in zip(new_im_rgb, list_class):
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

        list_polygons.append((contour_to_polygon(contours), class_name))
    
    return list_polygons