import cv2
import mediapipe as mp
import time
import variables as v
import numpy as np
from landmark_detection import (convert_landmarks_to_coordinates,
                                detect_centered_face,
                                detect_turned_head,
                                detect_raised_hand,
                                detect_good_posture,
                                detect_geometric_shape)
from kalman_tracking import initialize_kalman, correct_kalman, recalculate_histogram
from maquina_estados import *
from utils import play_alert
from UI import VisualEngine, UIColors
from enum import Enum, auto


class SystemMode(Enum):
    PASSWORD = auto()
    MONITOR = auto()
    PAUSED = auto()


def main(camera_index=0, width=1280, height=720, debug=False):
    print("Programa iniciado")

    password_fsm = PasswordFSM(stable_frames=5, cooldown_frames=5)
    ui = VisualEngine()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    last_mp_time = time.time()

    kf = None
    crop_hist = None
    track_window = None

    system_mode = SystemMode.PASSWORD
    prev_mode = SystemMode.PASSWORD
    estado_contraseña = State.WAITING_CENTERED

    reference_pose = None
    good_posture = True
    prev_good_posture = True
    waiting_frames = v.WAITING_FRAMES

    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return

    window_name = "Live Camera - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            guardar = []
            system_mode = SystemMode.PASSWORD
            prev_mode = SystemMode.PASSWORD
            while True:
                texts = []
                start = time.time()
                ret, frame = cap.read()
                px = py = None

                if not ret:
                    print("No frame received (the camera may have been disconnected).")
                    break
                frame = cv2.flip(frame, 1)
                if waiting_frames == v.WAITING_FRAMES or waiting_frames == 0:
                    triangle = detect_geometric_shape(frame)
                    waiting_frames = v.WAITING_FRAMES
                else:
                    triangle = False
                    waiting_frames -= 1

                ### --- PAUSAR ESTADO POR DETECCIÓN DE FLANCO---##
                if triangle:
                    waiting_frames -= 1
                    if system_mode == SystemMode.PAUSED:
                        system_mode = prev_mode
                        kf = None
                    elif system_mode == SystemMode.MONITOR:
                        prev_mode = system_mode
                        system_mode = SystemMode.PAUSED

                # -- Corrección de Kalman --#
                now = time.time()
                run_mp = (now - last_mp_time >= v.MP_INTERVAL) or (kf is None)

                if system_mode != SystemMode.PAUSED and run_mp:
                    results = pose.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    last_mp_time = now
                    if kf is not None and run_mp:
                        px = int(kf.statePost[0, 0])
                        py = int(kf.statePost[1, 0])
                    if results and results.pose_landmarks:
                        nose = results.pose_landmarks.landmark[v.NOSE]
                        cx, cy = convert_landmarks_to_coordinates(
                            width, height, nose.x, nose.y)
                        if kf is None:
                            kf, crop_hist, track_window = initialize_kalman(
                                cx, cy, frame)
                        else:
                            measurement = np.array([[cx], [cy]], np.float32)
                            kf.correct(measurement)

                            track_window = (
                                int(cx - v.HW), int(cy - v.HW), v.HW * 2, v.HW * 2)
                            roi = frame[max(
                                0, cy-v.HW):min(height, cy+v.HW), max(0, cx-v.HW):min(width, cx+v.HW)]
                            crop_hist = recalculate_histogram(roi)

                #### --- LÓGICA PRINCIPAL SEGÚN MODO-----#

                ### --------------- Process Detections & Tracking --------------- ###
                if system_mode != SystemMode.PAUSED:
                    if results and results.pose_landmarks:
                        if system_mode == SystemMode.PASSWORD:
                            centered = detect_centered_face(
                                results, width, height)
                            turned, direction = detect_turned_head(results)
                            raised, hand = detect_raised_hand(results)

                            gesture = get_gesture_from_flags(
                                turned, direction, raised, hand)
                            estado_contraseña, unlocked = password_fsm.step(
                                gesture, centered, results)

                            if unlocked:
                                reference_pose = password_fsm.reference_pose
                                system_mode = SystemMode.MONITOR

                        elif system_mode == SystemMode.MONITOR:
                            turned, direction = detect_turned_head(results)
                            if reference_pose is not None:
                                good_posture = detect_good_posture(
                                    results, reference_pose, turned)
                                if prev_good_posture == True and good_posture == False:
                                    play_alert()
                                prev_good_posture = good_posture
                            else:
                                good_posture = True
                            texts = [
                                "Postura correcta" if good_posture else "Corrige la postura"]

                if kf is not None and not run_mp:
                    px, py, track_window = correct_kalman(
                        frame, kf, crop_hist, track_window)

                    # Control de deriva MeanShift
                    kx, ky = kf.statePost[0, 0], kf.statePost[1, 0]
                    mx = track_window[0] + track_window[2] / 2
                    my = track_window[1] + track_window[3] / 2

                    if np.hypot(mx - kx, my - ky) > v.MAX_MS_DRIFT:
                        track_window = (
                            int(kx - v.HW), int(ky - v.HW), v.HW * 2, v.HW * 2)

                if kf is not None:
                    x, y, w_, h_ = track_window
                    draw_tracking = False
                    rect_color = (200, 200, 200)

                    if system_mode == SystemMode.PASSWORD:
                        draw_tracking = False
                        rect_color = (255, 0, 0)  # azul

                    elif system_mode == SystemMode.MONITOR:
                        draw_tracking = True
                        rect_color = (
                            0, 255, 0) if good_posture else (0, 0, 255)

                    if draw_tracking:
                        cv2.rectangle(
                            frame, (x, y), (x + w_, y + h_), rect_color, 2)
                        if px is not None and py is not None:
                            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

                if kf is not None and system_mode != SystemMode.PASSWORD and system_mode != SystemMode.PAUSED:
                    x, y, w_, h_ = track_window
                    t_color = UIColors.SUCCESS if good_posture else UIColors.DANGER
                    ui.draw_target_corners(frame, (x, y, w_, h_), t_color)

                ### --------------- Visualization --------------- ###
                end = time.time()
                fps = int(1 / max(end - start, 1e-6))
                panel_h = 100
                ui.draw_glass_panel(frame, 0, height - panel_h, width, panel_h)

                mode_color = UIColors.ACCENT
                if system_mode == SystemMode.MONITOR:
                    mode_color = UIColors.SUCCESS if good_posture else UIColors.DANGER
                elif system_mode == SystemMode.PAUSED:
                    mode_color = (150, 150, 150)
                frame = ui.draw_text_pro(
                    frame, f"MODE: {system_mode.name}", (40, height - 75), ui.font_bold, mode_color)

                # Barra de progreso solo en PASSWORD y no pausado
                if system_mode == SystemMode.PASSWORD:
                    ui.draw_password_progress(frame, password_fsm, x=450, y=40)
                    print(
                        f"El estado de la contraseña es: {estado_contraseña}")
                    if estado_contraseña == State.WAITING_CENTERED:
                        ui.draw_center_square(frame)

                status_text = "SYSTEM ACTIVE"
                if system_mode == SystemMode.PAUSED:
                    status_text = "PAUSED - SHOW CARD TO RESUME"
                elif texts:
                    status_text = texts[0].upper()

                frame = ui.draw_text_pro(
                    frame, status_text, (40, height - 35), ui.font_main, (200, 200, 200))
                frame = ui.draw_text_pro(
                    frame, f"FPS: {fps}", (width - 120, height - 35), ui.font_bold, UIColors.ACCENT)

                # Esqueleto en debug
                if debug:
                    results = pose.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if (results is not None) and results.pose_landmarks:
                        debug_str = f"H: {direction} | Hd: {hand} | C: {centered}"
                        frame = ui.draw_text_pro(
                            frame, debug_str, (width - 400, height - 75), ui.font_tiny, (0, 255, 0))
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(255, 255, 255), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(
                                color=(0, 191, 255), thickness=1)
                        )
                cv2.imshow(window_name, frame)

                ### --------------- Keyboard Control --------------- ###
                key = cv2.waitKey(1)
                if key == 32 and results is not None and results.pose_landmarks:
                    guardar.append([
                        results.pose_landmarks.landmark[v.NOSE],
                        results.pose_landmarks.landmark[v.RIGHT_EAR],
                        results.pose_landmarks.landmark[v.LEFT_EAR]])
                if key == ord('g'):
                    with open("cara_derecha", "w") as f:
                        f.write(
                            "nose_x,nose_y,nose_z,nose_v,right_ear_x,right_ear_y,right_ear_z,right_ear_v,left_ear_x,left_ear_y,left_ear_z,left_ear_v\n")
                        for item in guardar:
                            f.write(
                                f"{item[0].x},{item[0].y},{item[0].z},{item[0].visibility},{item[1].x},{item[1].y},{item[1].z},{item[1].visibility},{item[2].x},{item[2].y},{item[2].z},{item[2].visibility}\n")
                if key == ord('j'):
                    cv2.imwrite('image.png', frame)
                if key == ord('q') or key == 27:
                    break
    except Exception as error:
        print(f"An error occurred: {error}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # If the built-in webcam is not at index 0, change the first argument: main(1)
    print("Entrando en camara")
    main(camera_index=0, width=1280, height=720, debug=False)
