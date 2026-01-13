from utils import convert_landmarks_to_coordinates, read_data
from typing import Tuple
import variables as v
import cv2
import os


def segment_by_color(low_threshold, high_threshold, img):
    """Segment image by color.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_img, low_threshold, high_threshold)
    segmented_img = cv2.bitwise_and(img, img, mask=color_mask)
    return color_mask, segmented_img


def detect_geometric_shape(frame, low_threshold=v.PXL_HSV_LOW_MASK, high_threshold=v.PXL_HSV_HIGH_MASK):
    """Detects a triangle shape given the color thresholds.
    """
    mask, _ = segment_by_color(low_threshold, high_threshold, frame)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contornos, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 500:
            perimeter = cv2.arcLength(c, True)
            approx_points = cv2.approxPolyDP(c, 0.04 * perimeter, True)
            vertices = len(approx_points)
            if vertices == 3:
                return True
    return False


def obtain_ear_to_nose_ratio(df):
    """
    This function creates a dataframe to study the data extracted and obtain thresholds empirically to detect a turned head. It is used offline.
    """
    df['distance_left_to_right_ear'] = ((df['left_ear_x'] - df['right_ear_x']) ** 2 +
                                        (df['left_ear_y'] - df['right_ear_y']) ** 2 +
                                        (df['left_ear_z'] - df['right_ear_z']) ** 2) ** 0.5
    df['horizontal_nose_deltax'] = df['nose_x'] - \
        ((df['left_ear_x'] + df['right_ear_x'])/2)
    df['horizontal_ratio'] = df['horizontal_nose_deltax'] / \
        df['distance_left_to_right_ear']
    df['abs_horizontal_ratio'] = df['horizontal_ratio'].abs()

    return df


def detect_good_posture(results: dict, reference_pose: dict, turned: bool) -> bool:
    """Function that detects good posture. It takes into account the curvature of the back, the symetry of the shoulders,
    t and if the face is too close to the screen.
    Args:
        results (dict): mediapipe landmark detections
        turned (bool): Head is turned

    Returns:
        bool: Boolean that represents the detection of good posture
    """
    curved_back = detect_curved_back(results, reference_pose)
    shoulder_asymetry = detect_shoulder_asymetry(results, turned)
    close_face = detect_close_face(results)

    if not curved_back and not shoulder_asymetry and not close_face:
        return True
    return False


def detect_curved_back(results: dict, centered_pose: dict) -> bool:
    """Detects curved back comparing current position with straight pose.
    """
    current_left_ear = results.pose_landmarks.landmark[v.LEFT_EAR]
    current_left_shoulder = results.pose_landmarks.landmark[v.LEFT_SHOULDER]

    current_distance = abs(current_left_ear.y - current_left_shoulder.y)
    optimal_ear = centered_pose["LEFT_EAR"]
    optimal_shoulder = centered_pose["LEFT_SHOULDER"]
    optimal_distance = abs(optimal_ear.y - optimal_shoulder.y)

    threshold = 0.02
    if current_distance < (optimal_distance - threshold):
        return True
    return False


def detect_shoulder_asymetry(results: dict, turned: bool) -> bool:
    """
    Detects shoulder height asymmetry.
    """
    left_shoulder = results.pose_landmarks.landmark[v.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[v.RIGHT_SHOULDER]

    vertical_diff = abs(left_shoulder.y - right_shoulder.y)

    if not turned:
        return vertical_diff > 0.04
    else:
        return vertical_diff > 0.07


def detect_close_face(results: dict) -> bool:
    """
    Detects if the face is too close to the camera using inter-ear distance.
    """
    right_ear = results.pose_landmarks.landmark[v.RIGHT_EAR]
    left_ear = results.pose_landmarks.landmark[v.LEFT_EAR]

    ear_distance = (
        (left_ear.x - right_ear.x) ** 2 +
        (left_ear.y - right_ear.y) ** 2 +
        (left_ear.z - right_ear.z) ** 2
    ) ** 0.5

    return ear_distance > 0.16


def detect_turned_head(results: dict, threshold: float = 0.13) -> Tuple[bool, str]:
    """Function that detects turned head. Analysing 60 images of turned heads we obtain some statistics to set the threshold.
        mean - 0.257903 / std - 0.097363 / min - 0.133248 / max - 0.807061 / 50% - 0.232106

    Args:
        results (list): Results from mediapipe processing (results = pose.process(frame_rgb))
        threshold (float, optional): Threshold that detects turned head for horizontal_ratio. Defaults to 0.13.

    Returns:
        bool: Boolean that indicates if the head is turned
        str: Direction of the turned head ("right", "left", "forward")
    """
    nose = results.pose_landmarks.landmark[v.NOSE]
    right_ear = results.pose_landmarks.landmark[v.RIGHT_EAR]
    left_ear = results.pose_landmarks.landmark[v.LEFT_EAR]

    horizontal_nose_deltax = nose.x - ((left_ear.x + right_ear.x)/2)
    distance_left_to_right_ear = (((left_ear.x - right_ear.x) ** 2 +
                                   (left_ear.y - right_ear.y) ** 2 +
                                   (left_ear.z - right_ear.z) ** 2)) ** 0.5
    horizontal_ratio = horizontal_nose_deltax / distance_left_to_right_ear

    if abs(horizontal_ratio) > threshold:
        if horizontal_ratio > 0:
            return True, "right"
        else:
            return True, "left"
    else:
        return False, "forward"


def detect_raised_hand(results: dict) -> Tuple[bool, str]:
    """Function that detects raised hand. 

    Args:
        detected_points (list): Listed of Key points (Nose, Right_Wrist, Left_Wrist)
        threshold (float, optional): Threshold that detects raised hand for XXXXXXXXXXX. Defaults to 0.1.

    Returns:
        bool: Boolean that indicates if the hand is raised
        str: Direction of the raised hand ("right", "left", "none")
    """
    nose = results.pose_landmarks.landmark[v.NOSE]
    right_wrist = results.pose_landmarks.landmark[v.RIGHT_WRIST]
    left_wrist = results.pose_landmarks.landmark[v.LEFT_WRIST]

    if right_wrist.visibility >= v.VIS_THRESHOLD:
        if (right_wrist.y < nose.y + 0.1) or (right_wrist.y > nose.y - 0.1):
            return True, "left"  # Leave it like this
    if left_wrist.visibility >= v.VIS_THRESHOLD:
        if (left_wrist.y < nose.y + 0.1) or (left_wrist.y > nose.y - 0.1):
            return True, "right"
    return False, "none"


def detect_centered_face(results: dict, width: int, height: int) -> bool:
    """Function that detects centered face. 

    Args:
        results (dict): Results from mediapipe processing (results = pose.process(frame_rgb))
        threshold (float, optional): Threshold that detects centered face for XXXXXXXXXXX. Defaults to 0.1.

    Returns:
        bool: Boolean that indicates if the face is centered
    """
    nose = results.pose_landmarks.landmark[v.NOSE]
    right_ear = results.pose_landmarks.landmark[v.RIGHT_EAR]
    left_ear = results.pose_landmarks.landmark[v.LEFT_EAR]

    cx, cy = width // 2, height // 2
    nose_x, nose_y = convert_landmarks_to_coordinates(
        width, height, nose.x, nose.y)
    # Detect if nose is centered in rectangle
    if nose_x in range(cx - 50, cx + 50) and nose_y in range(cy - 50, cy + 50):
        # Detect if ears are inside the rectangle
        right_ear_x, _ = convert_landmarks_to_coordinates(
            width, height, right_ear.x, right_ear.y)
        left_ear_x, _ = convert_landmarks_to_coordinates(
            width, height, left_ear.x, left_ear.y)
        if right_ear_x <= cx + 130 and left_ear_x >= cx - 130:
            return True
    return False


def show_debug(width, height, frame, results, mp_drawing, mp_pose):
    """Function that draws in the video the detections and their processing.

    Args:
        width (int): Width of the frame
        height (int): Height of the frame
        frame (ndarray): Frame image from OpenCV
        results (dict): Processed landmarks from mediapipe
    """
    # Draw detected landmarks
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    # Draw debug information
    centered = detect_centered_face(results, width, height)
    turned, direction = detect_turned_head(results)
    raised, hand = detect_raised_hand(results)

    cx, cy = width // 2, height // 2
    cv2.rectangle(frame, (cx - 150, cy - 150),
                  (cx + 150, cy + 150),  (255, 0, 0), 5)
    xs, ys = convert_landmarks_to_coordinates(
        width, height, results.pose_landmarks.landmark[v.NOSE].x, results.pose_landmarks.landmark[v.NOSE].y)
    cv2.rectangle(frame, (xs - 150, ys - 150),
                  (xs + 150, ys + 150),  (0, 255, 0), 5)
    cv2.putText(
        frame,
        f"Turned: {turned}, Direction: {direction} / Raised: {raised}, Hand: {hand} / Centered: {centered}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return turned, direction, raised, hand, centered


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(base_path)

    ratio = "Ear_Nose"
    data_folder = os.path.join(
        parent_path, 'assets', 'Password_Parameters', f"Ratio_{ratio}")
    csv_files = [os.path.join(data_folder, file) for file in os.listdir(
        data_folder) if file.endswith('.csv')]

    df = read_data(csv_files)

    if ratio == "Ear_Nose":
        df_with_ratios = obtain_ear_to_nose_ratio(df)
        print(df_with_ratios)
        print(df_with_ratios[['abs_horizontal_ratio']].describe())
