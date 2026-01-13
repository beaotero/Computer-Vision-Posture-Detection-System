import cv2
import mediapipe as mp
import time


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Live Camera - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected).")
                break
            frame = cv2.flip(frame, 1)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                nombre = f"../assets/Calibration/captura_{int(time.time())}.png"
                cv2.imwrite(nombre, frame)
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # If the built-in webcam is not at index 0, change the first argument: main(1)
    main(camera_index=0, width=1280, height=720)
