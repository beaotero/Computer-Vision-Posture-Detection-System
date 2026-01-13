import cv2
from typing import List
import numpy as np
import imageio
import copy
import os


def load_images(filenames: List) -> List:
    """Load images using imageio

    Args:
        filenames (List): List of paths

    Returns:
        List: Loaded images
    """
    images = []
    for filename in filenames:
        try:
            img = imageio.v2.imread(filename)
            images.append(img)
        except Exception as e:
            print(f"Error loading image '{filename}': {e}")
            raise
    return images


def get_chessboard_points(chessboard_shape, dx, dy):
    """Function to get simulated chessboard points based of the characteristics of the board.

    Args:
        chessboard_shape (tuple): Number of inner corners per chessboard row and column (cols, rows)
        dx (float): Distance between corners in the x direction
        dy (float): Distance between corners in the y direction

    Returns:
        np.array: Array of 3D points representing the chessboard corners
    """
    cols, rows = chessboard_shape
    first_point = (0, 0, 0)
    chessboard_points = []
    for i in range(rows):
        for j in range(cols):
            chessboard_points.append(
                (first_point[0] + j * dx, first_point[1] + i * dy, 0))
    return np.array(chessboard_points, dtype=np.float32)


def show_image(img: np.array, img_name: str = "Image"):
    """Function to show an image in a pop-up window.

    Args:
        img (np.array): image to display
        img_name (str, optional): Name of the image window. Defaults to "Image".
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_image(output_folder: str, img_name: str, img: np.array):
    """Function to write image as a file

    Args:
        output_folder (str): Folder to store image
        img_name (str): Name of the image file
        img (np.array): Image to write
    """
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def draw_all_chessboard_corners(imgs: List[np.array], pattern_size: tuple, corners_refined: List[np.array], found: List[bool]):
    """Function to draw detected corners in an image

    Args:
        imgs (List[np.array]): images where to draw corners
        pattern_size (tuple): Number of inner corners per chessboard row and column (cols, rows)
        corners_refined (List[np.array]): Refined corner locations
        found (List[bool]): List indicating if corners were found in each image
    """
    for i in range(len(imgs)):
        try:
            im = cv2.drawChessboardCorners(
                imgs[i], pattern_size, corners_refined[i], found[i])
            show_image(im, f"Image {i}")
        except Exception as e:
            print(f"Error drawing corners on image {i}: {e}")


def calibration_pipeline(calibration_img_paths: List[str], pattern_size: tuple, debug: bool = False, square_size: int = 30):
    """Complete camera calibration pipeline

    Args:
        calibration_img_paths (List[str]): Paths to calibration images
        pattern_size (tuple): Number of inner corners per chessboard row and column (cols, rows)
        debug (bool, optional): If True, show images with detected corners. Defaults to False.
        square_size (int, optional): Size of a square in the chessboard (in any unit). Defaults to 30.

    Returns:
        tuple: Root mean squared error, camera matrix, distortion coefficients, and extrinsic parameters
    """
    imgs = load_images(calibration_img_paths)

    corners = [cv2.findChessboardCorners(
        image, pattern_size) for image in imgs]
    corners_copy = copy.deepcopy(corners)
    imgs_copy = copy.deepcopy(imgs)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    corners_refined = [cv2.cornerSubPix(i, cor[1], pattern_size, (-1, -1), criteria)
                       if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]
    chessboard_points = [get_chessboard_points(
        pattern_size, square_size, square_size) for _ in imgs_copy]

    if debug:
        draw_all_chessboard_corners(imgs_copy, pattern_size, corners_refined, [
                                    cor[0] for cor in corners_copy])

    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points, valid_corners, imgs_gray[0].shape, None, None)
    extrinsics = list(map(lambda rvec, tvec: np.hstack(
        (cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))
    return rms, intrinsics, dist_coeffs, extrinsics


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    calibration_dir = os.path.join(parent_dir, 'assets', 'Calibration')

    imgs_path = [os.path.join(calibration_dir, file)
                 for file in os.listdir(calibration_dir)]
    pattern_size = (9, 6)
    square_size = 24
    rms, intrinsics, dist_coeffs, extrinsics = calibration_pipeline(
        imgs_path, pattern_size, square_size=square_size)
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)
