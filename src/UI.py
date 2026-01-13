from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from maquina_estados import *


class UIColors:
    BG_DARK = (20, 20, 20, 160)
    ACCENT = (100, 145, 180)
    SUCCESS = (50, 215, 75)
    DANGER = (255, 69, 58)
    TEXT = (255, 255, 255)


class VisualEngine:
    def __init__(self):
        try:
            self.font_main = ImageFont.truetype("arial.ttf", 22)
            self.font_bold = ImageFont.truetype("arialbd.ttf", 26)
            self.font_tiny = ImageFont.truetype("arial.ttf", 16)
        except:
            self.font_main = self.font_bold = self.font_tiny = ImageFont.load_default()

    def draw_glass_panel(self, frame, x, y, w, h):
        """Draws a semi-transparent glass panel on the frame.
        """
        sub_face = frame[y:y+h, x:x+w]
        black_rect = np.zeros_like(sub_face)
        cv2.addWeighted(sub_face, 0.4, black_rect, 0.6, 0, sub_face)
        cv2.line(frame, (x, y), (x + w, y), (100, 100, 100), 1)

    def draw_text_pro(self, frame, text, pos, font, color=UIColors.TEXT):
        """Draws text on the frame using PIL for better font rendering.
        """
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_target_corners(self, frame, rect, color):
        """Draws corner lines around the rectangle of the area to track.
        """
        x, y, w, h = rect
        l = 25
        t = 3
        # Top Left
        cv2.line(frame, (x, y), (x + l, y), color, t)
        cv2.line(frame, (x, y), (x, y + l), color, t)
        # Top Right
        cv2.line(frame, (x+w, y), (x+w-l, y), color, t)
        cv2.line(frame, (x+w, y), (x+w, y+l), color, t)
        # Bottom Left
        cv2.line(frame, (x, y+h), (x+l, y+h), color, t)
        cv2.line(frame, (x, y+h), (x, y+h-l), color, t)
        # Bottom Right
        cv2.line(frame, (x+w, y+h), (x+w-l, y+h), color, t)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-l), color, t)

    def draw_password_progress(self, frame, password_fsm, x=30, y=30):
        """
        UI: progress bar.
        - Green: completed
        - Yellow: actual
        - Grey: pending
        """
        state_to_idx = {
            State.WAITING_CENTERED: 0,
            State.WAITING_LOOKING_LEFT: 1,
            State.WAITING_LOOKING_RIGHT: 2,
            State.WAITING_RAISING_RIGHT_HAND: 3,
            State.WAITING_RAISING_LEFT_HAND: 4,
            State.UNLOCKED: 5
        }
        idx = state_to_idx.get(password_fsm.state, 0)   # 0..5
        completed = max(0, min(idx, 5))
        # --- Layout ---
        steps = 5
        radius = 14
        gap = 78
        bar_th = 8

        title = "INTRODUCE PASSWORD"

        content_w = (steps - 1) * gap + 2 * radius
        panel_pad_x = 26
        panel_pad_y = 18
        title_h = 34

        panel_w = content_w + panel_pad_x * 2
        panel_h = title_h + 50 + panel_pad_y

        px1, py1 = x, y
        px2, py2 = x + panel_w, y + panel_h

        # --- Panel con transparencia ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (10, 10, 10), -1)
        alpha = 0.65
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Borde sutil
        cv2.rectangle(frame, (px1, py1), (px2, py2), (90, 90, 90), 1)

        # --- Título centrado, con sombra + línea acento ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.75
        title_th = 2

        (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_th)
        tx = px1 + (panel_w - tw) // 2
        ty = py1 + 26

        # sombra
        cv2.putText(frame, title, (tx + 2, ty + 2), font,
                    title_scale, (0, 0, 0), title_th + 2)
        # texto
        cv2.putText(frame, title, (tx, ty), font,
                    title_scale, (245, 245, 245), title_th)

        # línea/acento bajo el título
        line_y = py1 + title_h
        cv2.line(frame, (px1 + 18, line_y),
                 (px2 - 18, line_y), (60, 60, 60), 1)
        lower_top = py1 + title_h + 6
        lower_bottom = py2 - panel_pad_y
        cy = (lower_top + lower_bottom) // 2

        x_start = px1 + panel_pad_x + radius
        x_end = x_start + (steps - 1) * gap

        cv2.line(frame, (x_start, cy), (x_end, cy), (95, 95, 95), bar_th)
        if completed >= 1:
            prog_end = x_start + (min(completed, steps - 1)) * gap
            cv2.line(frame, (x_start, cy), (prog_end, cy), (0, 200, 0), bar_th)

        # nodos
        for i in range(steps):
            cx = x_start + i * gap

            if i < completed:
                color = (0, 200, 0)
                num_col = (15, 15, 15)
            elif i == completed and completed < 5:
                color = (0, 215, 255)
                num_col = (15, 15, 15)
            else:
                color = (150, 150, 150)
                num_col = (30, 30, 30)

            # círculo con borde
            cv2.circle(frame, (cx, cy), radius, color, -1)
            cv2.circle(frame, (cx, cy), radius, (20, 20, 20), 2)

            # número centrado
            n = str(i + 1)
            (nw, nh), _ = cv2.getTextSize(n, font, 0.6, 2)
            cv2.putText(frame, n, (cx - nw // 2, cy + nh // 2),
                        font, 0.6, num_col, 2)

    def draw_center_square(self, frame, size: int = 200, color=(128, 128, 128), thickness: int = 3, draw_crosshair: bool = True):
        """
        Dibuja un cuadrado centrado en la imagen.

        Args:
            frame: imagen BGR (np.array) de OpenCV.
            size: lado del cuadrado en píxeles.
            color: color BGR (por defecto azul).
            thickness: grosor del borde.
            draw_crosshair: si True, dibuja una cruz en el centro.

        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = size // 2

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w - 1)
        y2 = min(cy + half, h - 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if draw_crosshair:
            cross = max(10, size // 12)
            cv2.line(frame, (cx - cross, cy),
                     (cx + cross, cy), color, thickness)
            cv2.line(frame, (cx, cy - cross),
                     (cx, cy + cross), color, thickness)
        return
