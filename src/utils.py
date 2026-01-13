import pygame
import cv2
import os
import numpy as np
import pandas as pd
from maquina_estados import *


def convert_coordinates_to_landmarks(width: float, height: float, x: float, y: float):
    """
    Convert coordinates to landmark objects.
    """
    return x/width, y/height


def convert_landmarks_to_coordinates(width: float, height: float, xl: float, yl: float):
    """
    Convert landmark objects to coordinates.
    """
    return int(xl * width), int(yl * height)


def get_hsv_color_ranges(image: np.array):
    """Util to get low and high color threshold.
    """
    cv2.namedWindow('image')

    cv2.createTrackbar('HMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    cv2.setTrackbarPos('HMax', 'image', 255)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while (1):
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')

        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        cv2.imshow('image', output)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def nothing(x):
    pass


def read_data(list_paths: list) -> pd.DataFrame:
    """
    Read multiple CSV files and concatenate them into a single DataFrame.

    Args:
        list_paths (list): List of file paths to CSV files.
    """
    dataframes = [pd.read_csv(path) for path in list_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


pygame.mixer.init()


def play_alert():
    """Function that plays a specific downloaded sound to alert user.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        music_dir = os.path.join(parent_dir, 'assets', 'Music')
        sound_path = os.path.join(music_dir, 'alert_sound.mp3')

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")


if __name__ == '__main__':
    image = cv2.imread('image.png')

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("", hsv_image)
    get_hsv_color_ranges(hsv_image)
