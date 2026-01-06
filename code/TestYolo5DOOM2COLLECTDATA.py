import sys
import os
import time
import cv2
import torch
import numpy as np
from pynput import keyboard as pynput_keyboard
from vizdoom import DoomGame, Mode, ScreenResolution

# === Agregar yolov5 al path ===
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
# === Configuración de guardado ===
SAVE_PATH = "dataset"
FRAME_PATH = os.path.join(SAVE_PATH, "frames")
CSV_PATH = os.path.join(SAVE_PATH, "metadata.csv")

os.makedirs(FRAME_PATH, exist_ok=True)
data = []
frame_id = 0
frame_counter = 0
# === Configuración del modelo ===
MODEL_PATH = 'runs/train/doom_exp4/weights/best.pt'
DEVICE = select_device('cpu')
IMG_SIZE = 416

model = attempt_load(MODEL_PATH, device=DEVICE)
model.eval()
print("✅ Modelo YOLOv5 cargado correctamente")

# === Parámetros generales ===
SCREEN_WIDTH = 640
CENTER_X = SCREEN_WIDTH // 2
CENTER_TOLERANCE = 30
TARGET_FPS = 35
FRAME_DURATION = 1.0 / TARGET_FPS

# === Teclas presionadas con pynput ===
pressed_keys = set()

def on_press(key):
    try:
        pressed_keys.add(key.char.lower())
    except AttributeError:
        if key == pynput_keyboard.Key.space: pressed_keys.add("space")
        if key == pynput_keyboard.Key.left:  pressed_keys.add("left")
        if key == pynput_keyboard.Key.right: pressed_keys.add("right")

def on_release(key):
    try:
        pressed_keys.discard(key.char.lower())
    except AttributeError:
        if key == pynput_keyboard.Key.space: pressed_keys.discard("space")
        if key == pynput_keyboard.Key.left:  pressed_keys.discard("left")
        if key == pynput_keyboard.Key.right: pressed_keys.discard("right")

listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# === Dibuja las teclas presionadas en pantalla ===
def draw_pressed_keys(img):
    key_order = ["space", "e", "left", "right", "d", "a", "w", "s"]
    labels = {
        "space": "SPACE", "e": "E", "left": "←", "right": "→",
        "d": "D", "a": "A", "w": "W", "s": "S"
    }
    x, y = 10, 30
    for key in key_order:
        color = (0, 255, 0) if key in pressed_keys else (100, 100, 100)
        cv2.putText(img, f"[{labels[key]}]", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25

# === Inicializar VizDoom con nivel ===
def init_game(map_name):
    game = DoomGame()
    game.load_config("doom1.cfg")
    game.set_doom_scenario_path("DOOM2.WAD")
    game.set_doom_map(map_name)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)  # <- Modo con audio
    game.add_game_args("+snd_volume 1")       # Volumen general
    game.add_game_args("+snd_sfxvolume 1")    # Volumen de efectos
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    game.new_episode()
    print(f"🚀 Iniciando mapa: {map_name}")
    return game


# === Loop principal ===
map_index = 1
game = init_game(f"map{map_index:02d}")
last_time = time.time()

while True:
    if game.is_episode_finished():
        map_index += 1
        try:
            game.close()
            time.sleep(0.5)
            game = init_game(f"map{map_index:02d}")
        except Exception as e:
            print(f"❌ No se pudo cargar el mapa map{map_index:02d}: {e}")
            map_index = 1
            game = init_game("map01")

    actions = [0] * 9  # [attack, use, turn_left, turn_right, move_right, move_left, move_forward, move_backward]

    if "space" in pressed_keys: actions[0] = 1
    if "e"     in pressed_keys: actions[1] = 1
    if "left"  in pressed_keys: actions[2] = 1
    if "right" in pressed_keys: actions[3] = 1
    if "d"     in pressed_keys: actions[4] = 1
    if "a"     in pressed_keys: actions[5] = 1
    if "w"     in pressed_keys: actions[6] = 1
    if "s"     in pressed_keys: actions[7] = 1
    if "f" in pressed_keys: actions[8] = 1

    state = game.get_state()
    if state and state.screen_buffer is not None and state.screen_buffer.shape[0] == 3:
        frame = state.screen_buffer.transpose(1, 2, 0)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # YOLOv5
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(DEVICE) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            detections = non_max_suppression(pred, conf_thres=0.3)[0]

        manually_turning = "left" in pressed_keys or "right" in pressed_keys
        manually_attacking = "space" in pressed_keys

        closest_detection = None
        closest_offset = float('inf')

        if detections is not None and len(detections):
            detections[:, :4] = scale_boxes((IMG_SIZE, IMG_SIZE), detections[:, :4], img.shape).round()

            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) // 2
                offset = abs(center_x - CENTER_X)
                class_id = int(cls.item())
                label = f"{class_id} {conf:.2f}"

                # Definir colores por clase
                if class_id == 0:  # enemigo
                    color = (0, 255, 0)
                    if offset < closest_offset:
                        closest_offset = offset
                        closest_detection = (x1, y1, x2, y2, center_x)
                elif class_id == 1:  # barril
                    color = (0, 255, 255)
                elif class_id == 2:  # vida
                    color = (0, 200, 0)
                else:
                    color = (100, 100, 100)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if closest_detection:
            x1, y1, x2, y2, center_x = closest_detection
            offset = center_x - CENTER_X

            if not manually_turning:
                if abs(offset) > CENTER_TOLERANCE:
                    actions[2] = int(offset < 0)
                    actions[3] = int(offset > 0)

            if abs(offset) <= CENTER_TOLERANCE and not manually_attacking:
                actions[0] = 1  # disparar

        draw_pressed_keys(img)
        cv2.imshow("Freedoom AutoAim HUD", img)
        if cv2.waitKey(1) == 27: break  # ESC para salir
    # Ejecutar acción
    reward = game.make_action(actions)

    # Guardar imagen y datos cada 30 frames
    if frame_counter % 10 == 0:
        state = game.get_state()
        if state and state.screen_buffer is not None and state.screen_buffer.shape[0] == 3:
            frame = state.screen_buffer.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C), RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_filename = f"{frame_id}.png"
            cv2.imwrite(os.path.join(FRAME_PATH, frame_filename), frame_bgr)

            # Guardar datos en lista
            data.append({
                "frame": frame_filename,
                "action_attack": actions[0],
                "action_use": actions[1],
                "action_turn_left": actions[2],
                "action_turn_right": actions[3],
                "action_move_right": actions[4],
                "action_move_left": actions[5],
                "action_move_forward": actions[6],
                "action_move_backward": actions[7],
                "reward": reward
            })

            print(f"[{frame_id}] ✅ Frame guardado con acciones: {actions}")
            frame_id += 1
        else:
            print(f"[{frame_id}] ❌ Frame no válido, omitido")

    frame_counter += 1
    time.sleep(0.01)
    game.make_action(actions)

    elapsed = time.time() - last_time
    time.sleep(max(0, FRAME_DURATION - elapsed))
    last_time = time.time()

game.close()
cv2.destroyAllWindows()
