import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    num = math.log(1 - prob_success)
    denom = math.log(1 - (ind_prob_correct ** sample_size))

    num_samples = math.ceil(num / denom) # round up so that we have at least as many samples as we need

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)

def ransac_homography(
    points_a: np.ndarray, points_b: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################

    # Define Params to tune
    num_iterations = calculate_num_ransac_iterations(0.998, 4, 0.1)
    inlier_thresh = 5 # pixels for inlier threshold (using homography, is point B within 10 pixels of point H*A)?

    print("num_iterations: ", num_iterations)

    # Loop through RANSAC iterations
    best_H = np.zeros((3, 3))
    best_inliers_a = np.zeros((0, 2))
    best_inliers_b = np.zeros((0, 2))
    best_num_inliers = 0

    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(points_a), 4, replace=False)
        sample_a = points_a[sample_indices]
        sample_b = points_b[sample_indices]

        H, _ = cv2.findHomography(sample_a, sample_b, 0)

        if H is None: # docs say it can be none if the solve fails
            continue

        # prepare A for transformation
        A_homogenous = np.vstack([points_a.T, np.ones((len(points_a)))])
        transformed_A = H@A_homogenous # this should spit out an Nx3 matrix
        transformed_A = transformed_A.T # transpose to get Nx3
        transformed_A = transformed_A[:,:2] / transformed_A[:, 2:3] # divide by last col and remove to move out of homogenous


        # Compute distances between transformed A and B
        distances = np.linalg.norm(transformed_A - points_b, axis=1) # Compare each point
        inliers = distances < inlier_thresh # boolean mask
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_H = H
            best_inliers_a = points_a[inliers]
            best_inliers_b = points_b[inliers]

    inliers_a = best_inliers_a
    inliers_b = best_inliers_b

    print(best_H)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_H, inliers_a, inliers_b
