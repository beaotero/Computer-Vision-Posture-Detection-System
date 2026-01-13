import cv2
import numpy as np
import variables as v


def initialize_kalman(cx, cy, frame):
    """Initialize Kalman filter and get HSV histogram of the region we want to track.

    Args:
        cx (float): Coordinate X of Kalman's inital position
        cy (float): Coordinate Y of Kalman's inital position
        frame (np.array): video frame

    Returns:
        Tuple: Kalman filter, HSV histogram, track window
    """
    kf = cv2.KalmanFilter(4, 2)
    dt = 1 / 20
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1

    kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    x = int(cx - v.HW)
    y = int(cy - v.HW)
    w = h = v.HW * 2
    track_window = (x, y, w, h)

    crop = frame[y: y + h, x: x + w].copy()
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_crop, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
    )
    crop_hist = cv2.calcHist([hsv_crop], [0], mask=mask,
                             histSize=[32], ranges=[0, 180])
    cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

    return kf, crop_hist, track_window


def correct_kalman(frame, kf, crop_hist, track_window):
    """Corrects Kalman prediction given a new frame.

    Args:
        frame (np.array): Video frame
        kf (cv2.KalmanFilter): Kalman Filter
        crop_hist (MatLike): HSV Histogram
        track_window (Tuple): Coordinates of track window

    Returns:
        Tuple: center x, center y, updated track window
    """
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hsv], [0], crop_hist, [0, 180], 1)

    _, track_window = cv2.meanShift(backproj, track_window, term_crit)
    x, y, w, h = track_window
    cx = x + w / 2
    cy = y + h / 2
    prediction = kf.predict()
    measurement = np.array([[cx], [cy]], np.float32)
    kf.correct(measurement)

    px, py = int(prediction[0]), int(prediction[1])
    return px, py, track_window


def recalculate_histogram(roi):
    """Function that recalculates de histogram for MeanShift.

    Args:
        roi (array): Rectangle region of interest

    Returns:
        MatLike: normalized HSV histogram
    """
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
    )
    crop_hist = cv2.calcHist([hsv_roi], [0], mask, [32], [0, 180])
    cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)
    return crop_hist
