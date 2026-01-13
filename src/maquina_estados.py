from enum import Enum, auto
import variables as v


class Gesture(Enum):
    NONE = auto()
    LOOK_LEFT = auto()
    LOOK_RIGHT = auto()
    RAISE_LEFT_HAND = auto()
    RAISE_RIGHT_HAND = auto()


class State(Enum):
    WAITING_CENTERED = 0
    WAITING_LOOKING_LEFT = 1
    WAITING_LOOKING_RIGHT = 2
    WAITING_RAISING_LEFT_HAND = 3
    WAITING_RAISING_RIGHT_HAND = 4
    UNLOCKED = 5


def capture_optimal_posture(results: dict):
    """Function that captures the main posture landmarks in a given moment.
    Args:
        results (dict): mediapipe landmark detections

    Returns:
        dict: Dictionary containing the main landmarks
    """
    return {"NOSE": results.pose_landmarks.landmark[v.NOSE],
            "RIGHT_EAR": results.pose_landmarks.landmark[v.RIGHT_EAR],
            "LEFT_EAR": results.pose_landmarks.landmark[v.LEFT_EAR],
            "LEFT_SHOULDER": results.pose_landmarks.landmark[v.LEFT_SHOULDER], }


def get_gesture_from_flags(turned, direction, raised, hand) -> Gesture:
    if turned and direction == "left":
        return Gesture.LOOK_LEFT
    if turned and direction == "right":
        return Gesture.LOOK_RIGHT
    if raised and hand == "right":
        return Gesture.RAISE_RIGHT_HAND
    if raised and hand == "left":
        return Gesture.RAISE_LEFT_HAND
    return Gesture.NONE


class PasswordFSM:
    def __init__(self, stable_frames=5, cooldown_frames=5):
        self.PASSWORD = [
            0,
            Gesture.LOOK_LEFT,
            Gesture.LOOK_RIGHT,
            Gesture.RAISE_RIGHT_HAND,
            Gesture.RAISE_LEFT_HAND,
        ]

        self.stable_frames = stable_frames
        self.cooldown_frames = cooldown_frames

        self.reference_pose = None

        self.reset()

    def reset(self):
        self.state = State.WAITING_CENTERED
        self._stable_count = 0
        self._cooldown = 0
        self._last_seen = None
        self.reference_pose = None

    # ---------------------------------------------------
    def step(self, gesture: Gesture, centered: bool, results):
        """
        Llamar CADA FRAME con:
        - gesture: gesto actual (o Gesture.NONE)

        Devuelve:
        unlocked (bool), lista_de_textos_para_pintar_en_pantalla
        """
        # Una vez desbloqueado, ya no cambia
        if self.state == State.UNLOCKED:
            return self.state, True

        # Si acaba de centrarse
        if self.state == State.WAITING_CENTERED and centered:
            self.state = State.WAITING_LOOKING_LEFT
            self.reference_pose = capture_optimal_posture(results)

        elif self.state == State.WAITING_LOOKING_LEFT:
            if gesture == Gesture.LOOK_LEFT:
                self.state = State.WAITING_LOOKING_RIGHT
            elif gesture != Gesture.NONE:
                self.reset()

        elif self.state == State.WAITING_LOOKING_RIGHT:
            if gesture == Gesture.LOOK_RIGHT:
                self.state = State.WAITING_RAISING_RIGHT_HAND
            elif gesture != Gesture.LOOK_LEFT and gesture != Gesture.NONE:
                self.reset()

        elif self.state == State.WAITING_RAISING_RIGHT_HAND:
            if gesture == Gesture.RAISE_RIGHT_HAND:
                self.state = State.WAITING_RAISING_LEFT_HAND
            elif gesture != Gesture.LOOK_RIGHT and gesture != Gesture.NONE:
                self.reset()

        elif self.state == State.WAITING_RAISING_LEFT_HAND:
            if gesture == Gesture.RAISE_LEFT_HAND:
                self.state = State.UNLOCKED

            elif gesture != Gesture.RAISE_RIGHT_HAND and gesture != Gesture.NONE:
                self.reset()

        if self.state == State.UNLOCKED:
            return self.state, True

        else:
            return self.state, False
