# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import os
import matplotlib.pyplot as plt


# Code inspired from https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/ and https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Path('C:/Program Files\ Tesseract-OCR\ tesseract').as_posix()
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

def decode_predictions(scores, geometry, min_confidence):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < min_confidence:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# round to the nearest multiple
def round_down(num, divisor):
    return num - (num%divisor)

def get_OCR_predictions(image_file, visualization=False):
	# load the input image and grab the image dimensions
	image = cv2.imread(image_file)
	orig = image.copy()
	(origH, origW) = image.shape[:2]
	# set the new width and height and then determine the ratio in change
	# for both the width and height
	#(newW, newH) = (2 * origW, 2*origH) #(args["width"], args["height"])
	(newW, newH) = (origW, origH)
	(newW, newH) = (round_down(newW, 32), round_down(newH, 32))
	rW = origW / float(newW)
	rH = origH / float(newH)
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	# define the two output layer names for the EAST detector model that
	# we are interested in -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	min_conf = 0.05
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry, min_conf)
	boxes = non_max_suppression(np.array(rects), probs=confidences)


	# initialize the list of results

	padding_value = 0.25
	results = []
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		# in order to obtain a better OCR of the text we can potentially
		# apply a bit of padding surrounding the bounding box -- here we
		# are computing the deltas in both the x and y directions
		dX = int((endX - startX) * padding_value)
		dY = int((endY - startY) * padding_value)
		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))
		# extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]
		# in order to apply Tesseract v4 to OCR text we must supply
		# (1) a language, (2) an OEM flag of 4, indicating that the we
		# wish to use the LSTM neural net model for OCR, and finally
		# (3) an OEM value, in this case, 7 which implies that we are
		# treating the ROI as a single line of text
		config = ("--oem 1 --psm 11") # --l eng
		text = pytesseract.image_to_string(roi, config=config, lang="eng+kor+fra+chi_sim+chi_sim_vert+chi_tra+chi_tra_vert+jpn+jpn_vert+deu")
		# add the bounding box coordinates and OCR'd text to the list
		# of results
		results.append(((startX, startY, endX, endY), text))

	# sort the results bounding box coordinates from top to bottom
	results = sorted(results, key=lambda r:r[0][1])
	output = orig.copy()
	# loop over the results

	list_output = []
	for ((startX, startY, endX, endY), text) in results:
		# display the text OCR'd by Tesseract
		#print("OCR TEXT")
		#print("========")
		#print("{}\n".format(text))
		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw the text and a bounding box surrounding
		# the text region of the input image
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		if text != '':
			#output = orig.copy()
			cv2.rectangle(output, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(output, text, (startX, startY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
			list_output.append(((startX, startY, endX, endY), text))

	if visualization:
		plt.figure()
		plt.imshow(output)

	return list_output


def visualize_OCR_predictions(list_prediction, image_file, list_pred=False):
	image = cv2.imread(image_file, 1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	for ((startX, startY, endX, endY), text) in list_prediction:
		if list_pred:
			print("{}\n".format(text))
	
		cv2.rectangle(image, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
		cv2.putText(image, text, (startX, startY - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	plt.figure()
	plt.imshow(image)


####### Handle misrecgontions (mostly misspellings)


# Methods for correction

### Handling misspellings


# MEthod 1

import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Method 2

"""
https://www.kaggle.com/yk1598/symspell-spell-corrector/code

The spell checker has been entirely ripped off of this script by Serg Lavrikov(rumbok):
https://www.kaggle.com/rumbok/ridge-lb-0-41944

do check it out, its a work of art.

caveat: script consumes a lot of memory but is much faster than Norvig's spell checker (1 million times)
http://blog.faroo.com/2015/03/24/fast-approximate-string-matching-with-large-edit-distances/
"""

import re, random
import spacy
nlp = spacy.load('en_core_web_sm')

to_sample = False # if you're impatient switch this flag

def spacy_tokenize(text):
    return [token.text for token in nlp.tokenizer(text)]
    
def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.
    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/
    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.
    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.
    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.
    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2
    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary


    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]
        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None

def spell_corrector(word_list, words_d, ss) -> str:
    result_list = []
    for word in word_list:
        if word not in words_d:
            suggestion = ss.best_word(word, silent=True)
            if suggestion is not None:
                result_list.append(suggestion)
        else:
            result_list.append(word)
            
    return " ".join(result_list)


def prepare_corrector(dictionary_file, ss):
    
      
    
    # fetch english words dictionary
    #with open('479k-english-words/english_words_479k.txt') as f:
    with open(dictionary_file) as f:

        words = f.readlines()
    eng_words = [word.strip() for word in words]
    
    # Print some examples
    print(eng_words[:5])

    print('Total english words: {}'.format(len(eng_words)))
    
    print('create symspell dict...')
    
    if to_sample:
        # sampling from list for kernel runtime
        sample_idxs = random.sample(range(len(eng_words)), 100)
        eng_words = [eng_words[i] for i in sorted(sample_idxs)] # make sure our sample misspell is in there
    
    all_words_list = list(set(eng_words))
    silence = ss.create_dictionary_from_arr(all_words_list, token_pattern=r'.+')
    
    # create a dictionary of rightly spelled words for lookup
    words_dict = {k: 0 for k in all_words_list}
    
    return words_dict
    
def correction_prediction(sample_text, words_dict, ss):
    tokens = spacy_tokenize(sample_text)
    
    #print('run spell checker...')
    #print()
    #print('original text: ' + sample_text)
    #print()
    correct_text = spell_corrector(tokens, words_dict, ss)
    return correct_text
"""    
if __name__ == '__main__':
    
    words_dict = prepare_corrector('479k-english-words/english_words_479k.txt')
    sample_text = 'to infifity and byond'
    correct_text = correction_prediction(sample_text, words_dict)
    print('corrected text: ' + correct_text)

    print('Done.')    
"""

# build symspell tree 


def prepareDictForMisspellings():
    ss = SymSpell(max_edit_distance=2)
    words_dict = prepare_corrector('479k-english-words/english_words_479k.txt', ss)
    return ss, words_dict

def accountForMisspellings(list_output_OCR, words_dict, ss):
    processed_output_OCR = []
    for output_OCR in list_output_OCR:
        position = output_OCR[0]
        list_processed = list(set([output_OCR[1], correction(output_OCR[1]), correction(output_OCR[1].lower()), correction_prediction(output_OCR[1], words_dict, ss), correction_prediction(output_OCR[1].lower(), words_dict, ss)]))
        #print(correction(output_OCR[1]))
        #print(correction_prediction(output_OCR[1], words_dict, ss))
        processed_output_OCR.append((position, list_processed))
    return processed_output_OCR


## Named Entity Recognition


from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize 
import nltk
from nltk.tag import pos_tag
from nltk.tree import Tree

def formatted_entities(classified_paragraphs_list):
    entities = []

    for classified_paragraph in classified_paragraphs_list:
        for entry in classified_paragraph:
            entry_value = entry[0]
            entry_type = entry[1]
            entities.append((entry_value, entry_type))
            #if entry_type == 'LOCATION': 
            #    entities.append(entry_value) 
    return entities

def NERWithOldStanford(input_sample):
    java_path = "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe" #"C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
               'stanford-ner.jar',
               encoding='utf-8')
    tokenized_text = word_tokenize(input_sample)     
    classified_paragraphs_list = tagger.tag_sents([tokenized_text]) 
    formatted_result = formatted_entities(classified_paragraphs_list) 
    return formatted_result

nltk.download('maxent_ne_chunker')
nltk.download('words')
def NERNewVersion(input_sample):
    ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(input_sample)))
    continuous_chunk = []
    current_chunk = []
    entity_type = []
    for i in ne_tree:
        if type(i) == Tree:
            #print(i.leaves())
            #print(i.label())
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            entity_type.append(i.label())
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            named_entity_type =  " ".join(entity_type)

            if named_entity not in continuous_chunk:
                continuous_chunk.append((named_entity, named_entity_type))
                current_chunk = []
                entity_type = []
        else:
            continue

    return continuous_chunk

