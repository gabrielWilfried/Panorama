import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

try:
     # Step 1: Load the images to be stitched together.
    images = []
    num_images = int(input("Enter the number of images to stitch: "))

    # Open a file dialog to select the images
    Tk().withdraw()  
    filenames = askopenfilenames(title="Select Images", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    filenames = list(filenames)[:num_images] 

    for filename in filenames:
        image = cv2.imread(filename)
        if image is None:
            print("Failed to load the image. Please try again.")
            continue
        images.append(image)
    # create a feature detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors for each image
    keypoints = []
    descriptors = []
    for image in images:
        keypoint, descriptor = orb.detectAndCompute(image, None)
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    # Match feature points between the images.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors[0], descriptors[1])
    matches = sorted(matches, key=lambda x: x.distance)

    # Estimate the homography matrix between each pair of images.
    src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the images using the homography matrices to align them.
    result = cv2.warpPerspective(images[0], homography, (images[0].shape[1] + images[1].shape[1], images[0].shape[0]))
    result[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

    # Step 6: Blend the warped images together to create a panorama.
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    panorama = cv2.bitwise_and(result, result, mask=mask)

    # Step 7: Save the panorama to a file.
    cv2.imwrite("panorama.jpg", panorama)
    print("Panorama successfully created and saved as 'panorama.jpg'")
except Exception as e:
    print("An error occurred:", str(e))