# Import packages
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
import coord
import pytesseract

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'plate_frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'plate-detection.pbtxt')
# PATH_TO_IMAGE = os.path.join(CWD_PATH ,IMAGE_NAME)

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def plate_recog(img):
    image = img
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    img=image
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=2,
      min_score_thresh=0.90
      )

    cv2.imshow('plates',image)
    cv2.waitKey(0)

    # Getting coordinates of detected images
    coordinates = coord.return_coordinates(img,
                                           np.squeeze(boxes),
                                           np.squeeze(classes).astype(np.int32),
                                           np.squeeze(scores),
                                           category_index,
                                           use_normalized_coordinates=True,
                                           line_thickness=0.5,
                                           min_score_thresh=0.90)

    image_classes_list = []
    for index, value in enumerate(classes[0]):
        if scores[0, index] >= 0.90:
            image_classes_list.append(category_index.get(value)['name'])

    j = 0
    cropped_img_lists = []
    for coordinate in coordinates:
        (y1, y2, x1, x2, acc) = coordinate
        height = y2 - y1
        width = x2 - x1
        crop = image[y1:y1 + height, x1:x1 + width]
        cropped_img_lists.append(crop)
        cv2.imshow("licences plate", cropped_img_lists[j])
        cv2.waitKey(0)
        # To do: Recognize licence plate and generate challan
        #img_rgb = cv2.cvtColor(cropped_img_lists[j], cv2.COLOR_BGR2RGB)
        #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        #print(" eChallan genearted for:- " + pytesseract.image_to_string(img_rgb))
        j = j + 1
