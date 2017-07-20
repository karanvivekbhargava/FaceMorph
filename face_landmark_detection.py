#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import cv2
import numpy as np
import sys
import os
import dlib
import glob
from skimage import io
from imutils import face_utils
import imutils
from scipy.spatial import Delaunay;


# if len(sys.argv) != 3:
#     print(
#         "Give the path to the trained shape predictor model as the first "
#         "argument and then the directory containing the facial images.\n"
#         "For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#         "You can download a trained facial shape predictor from:\n"
#         "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#     exit()

predictor_path = 'shape_predictor_68_face_landmarks.dat' #sys.argv[1]
faces_folder_path = 'Faces'; #sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = io.imread(f)

#     win.clear_overlay()
#     win.set_image(img)

#     # Ask the detector to find the bounding boxes of each face. The 1 in the
#     # second argument indicates that we should upsample the image 1 time. This
#     # will make everything bigger and allow us to detect more faces.
#     dets = detector(img, 1)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#         # Get the landmarks/parts for the face in box d.
#         shape = predictor(img, d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
#                                                   shape.part(1)))
#         # Draw the face landmarks on the screen.
#         win.add_overlay(shape)

#     win.add_overlay(dets)
#     dlib.hit_enter_to_continue()

def get_facial_landmarks(filename):
    image = io.imread(filename);
    # detect face(s)
    dets = detector(image, 1);
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(image, d);
        shape = face_utils.shape_to_np(shape);

    # # loop over the face parts individually
    # for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    #     # clone the original image so we can draw on it, then
    #     # display the name of the face part on the image
    #     clone = image.copy()
    #     cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #         0.7, (0, 0, 255), 2)
 
    #     # loop over the subset of facial landmarks, drawing the
    #     # specific face part
    #     for (x, y) in shape[i:j]:
    #         cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
    #     # extract the ROI of the face region as a separate image
    #     (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    #     roi = image[y:y + h, x:x + w]
    #     roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
 
    #     # # show the particular face part
    #     # cv2.imshow("ROI", roi)
    #     # cv2.imshow("Image", clone)
    #     # cv2.waitKey(0)
 
    # # visualize all facial landmarks with a transparent overlay
    # output = face_utils.visualize_facial_landmarks(image, shape)
    # cv2.imshow("Image", output)
    # cv2.waitKey(0)

    return shape;