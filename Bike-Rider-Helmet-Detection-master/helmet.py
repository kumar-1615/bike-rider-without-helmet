# Import packages
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import coord
import plate

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

'''def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [
        cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
        for im in im_list]
    return cv2.hconcat(im_list_resize)'''

def helmet_detector():
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    IMAGE_NAME = 'validate'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'helmet_frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'helmet-detection.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 2

    # Load the label map.
    # Label maps map indices to category names
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    PATH_TO_TEST_IMAGES_DIR = os.path.join(CWD_PATH, IMAGE_NAME)
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'validate_image{}.jpg'.format(i)) for i in range(0, 10)]

    for image_path in TEST_IMAGE_PATHS:
        image = cv2.imread(image_path)
        #image = cv2.resize(image, (720, 480))
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        img=image
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.90)

        # All the results have been drawn on image. Now display the image.
        cv2.imshow('Riders detection', image)
        cv2.waitKey(0)

        coordinates = coord.return_coordinates(img,
                                                     np.squeeze(boxes),
                                                     np.squeeze(classes).astype(np.int32),
                                                     np.squeeze(scores),
                                                     category_index,
                                                     use_normalized_coordinates=True,
                                                     line_thickness=1,
                                                     min_score_thresh=0.90)

        image_classes_list = []
        for index, value in enumerate(classes[0]):
            if scores[0, index] >= 0.90:
                image_classes_list.append(category_index.get(value)['name'])

        i = 0
        cropped_img_lists = []
        no_helmet_list = []
        for coordinate in coordinates:
            (y1, y2, x1, x2, acc) = coordinate
            height = y2 - y1
            width = x2 - x1
            crop = img[y1:y1 + height, x1:x1 + width]
            cropped_img_lists.append(crop)
            if image_classes_list[i] == 'No Helmet':
                no_helmet_list.append(cropped_img_lists[i])
            i = i + 1

        if len(no_helmet_list)==0:
            print("Rider are with helmet")
        else:
            for helmet in no_helmet_list:
                print("Detecting licence plate...")
                plate.plate_recog(helmet)

        '''if len(no_helmet_list) == 1:
            print("Rider is without helmet")
            print("Detecting licence plate...")
            #plate.plate_recognition(no_helmet_list[0], image_list)
            plate.plate_recog(no_helmet_list[0])
        elif len(no_helmet_list) > 1:
            print("Riders are without helmet")
            print("Detecting licence plate...")
            
            im_h_resize = hconcat_resize_min(no_helmet_list)
            #plate.plate_recognition(im_h_resize, image_list)
            plate.plate_recog(im_h_resize)
        else:
            print("Riders are with helmet")'''

            # Press any key to close the image
        cv2.waitKey(0)

        # Clean up
        cv2.destroyAllWindows()



helmet_detector()
