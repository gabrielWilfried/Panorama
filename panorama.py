import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

try:
    # Step 1: Load the images to be stitched together
    num_images = 0
   

    while num_images <= 0:
        try:
            num_images = int(input("Enter the number of images to stitch: "))
            if num_images <= 0:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer value.")

    # Open a file dialog to select the images
    Tk().withdraw()
    filenames = askopenfilenames(
        title="Select Images",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")),
    )
    filenames = list(filenames)[:num_images]

    if len(filenames) < num_images:
        raise ValueError("Insufficient number of images selected.")

    images = []
    stitched_image = None
    orb = cv2.ORB_create()

    # Load and stitch the images
    for filename in filenames:
        image = cv2.imread(filename)
        if image is None:
            print(f"Failed to load the image '{filename}'. Please make sure it is a valid image file.")
            continue


        if stitched_image is None:
            stitched_image = image
        else:
            # Find keypoint matches between the current image and the stitched image
            gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)

            keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate the homography matrix
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Warp the current image to align with the stitched image
            warped_image = cv2.warpPerspective(image, homography, (stitched_image.shape[1] + image.shape[1], stitched_image.shape[0]))
            warped_image[0:stitched_image.shape[0], 0:stitched_image.shape[1]] = stitched_image

            stitched_image = warped_image

    # Save the stitched panorama to a file
    cv2.imwrite("panorama.jpg", stitched_image)
    print("Panorama successfully created and saved as 'panorama.jpg'")

except Exception as e:
    print("An error occurred:", str(e))