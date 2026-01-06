import sys
import os
import time
import cv2
import torch
import numpy as np
from pynput import keyboard as pynput_keyboard
from vizdoom import DoomGame, Mode, ScreenResolution

# === Agregar yolov5 al path ===
BASE_DIR = os.path.dirname(__file__)
YOLOV5_PATH = os.path.join(BASE_DIR, 'yolov5')
sys.path.insert(0, YOLOV5_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# === Configuración de modelos ===
MODEL_ITEMS_PATH = os.path.join(YOLOV5_PATH, 'runs', 'train', 'doom_exp5', 'weights', 'best.pt')
MODEL_ENEMIES_PATH = os.path.join(YOLOV5_PATH, 'runs', 'train', 'doom_exp9', 'weights', 'best.pt')

DEVICE = select_device('cpu')
IMG_SIZE = 416

model_items = attempt_load(MODEL_ITEMS_PATH, device=DEVICE)
model_enemies = attempt_load(MODEL_ENEMIES_PATH, device=DEVICE)
model_items.eval()
model_enemies.eval()
print("Modelos YOLOv5 cargados correctamente")

# === Parámetros generales ===
SCREEN_WIDTH = 640
CENTER_X = SCREEN_WIDTH // 2
CENTER_TOLERANCE = 30
TARGET_FPS = 144
FRAME_DURATION = 1.0 / TARGET_FPS

pressed_keys = set()
auto_assist_enabled = False

def on_press(key):
    global auto_assist_enabled
    try:
        k = key.char.lower()
        pressed_keys.add(k)
        if k == 'l':
            auto_assist_enabled = not auto_assist_enabled
            print(f"[Ayuda {'activada' if auto_assist_enabled else 'desactivada'}]")
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

def draw_pressed_keys(img):
    key_order = ["space", "e", "left", "right", "d", "a", "w", "s"]
    labels = {
        "space": "SPACE", "e": "E", "left": "<-", "right": "->",
        "d": "D", "a": "A", "w": "W", "s": "S"
    }
    x, y = 10, 30
    for key in key_order:
        color = (0, 255, 0) if key in pressed_keys else (100, 100, 100)
        cv2.putText(img, f"[{labels[key]}]", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25

    # Mostrar texto "L para ayuda"
    text = "L para ayuda (ON)" if auto_assist_enabled else "L para ayuda (OFF)"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = img.shape[1] - text_size[0] - 10
    text_y = 25
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def init_game(map_name):
    game = DoomGame()
    game.load_config(os.path.join(BASE_DIR, "doom1.cfg"))
    game.set_doom_scenario_path(os.path.join(BASE_DIR, "DOOM2.WAD"))
    game.set_doom_map(map_name)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.add_game_args("+snd_volume 1")
    game.add_game_args("+snd_sfxvolume 1")
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    game.new_episode()
    print(f" Iniciando mapa: {map_name}")
    return game

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
            print(f"No se pudo cargar el mapa map{map_index:02d}: {e}")
            map_index = 1
            game = init_game("map01")

    actions = [0] * 9
    if "space" in pressed_keys: actions[0] = 1
    if "e"     in pressed_keys: actions[1] = 1
    if "left"  in pressed_keys: actions[2] = 1
    if "right" in pressed_keys: actions[3] = 1
    if "d"     in pressed_keys: actions[4] = 1
    if "a"     in pressed_keys: actions[5] = 1
    if "w"     in pressed_keys: actions[6] = 1
    if "s"     in pressed_keys: actions[7] = 1
    if "f"     in pressed_keys: actions[8] = 1

    state = game.get_state()
    if state and state.screen_buffer is not None and state.screen_buffer.shape[0] == 3:
        frame = state.screen_buffer.transpose(1, 2, 0)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img_letterboxed = letterbox(img, new_shape=IMG_SIZE)[0]
        img_tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).float().to(DEVICE) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        detections_enemies = None
        detections_items = None
        closest_detection = None
        manually_turning = "left" in pressed_keys or "right" in pressed_keys
        manually_attacking = "space" in pressed_keys
        closest_offset = float('inf')  

        if auto_assist_enabled:
            with torch.no_grad():
                pred_items = model_items(img_tensor)[0]
                pred_enemies = model_enemies(img_tensor)[0]

            detections_items = non_max_suppression(pred_items, conf_thres=0.4)[0]
            detections_enemies = non_max_suppression(pred_enemies, conf_thres=0.4)[0]

            if detections_enemies is not None and len(detections_enemies):
                detections_enemies[:, :4] = scale_boxes(img_letterboxed.shape[:2], detections_enemies[:, :4], img.shape).round()
                for *xyxy, conf, cls in detections_enemies:
                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x = (x1 + x2) // 2
                    offset = abs(center_x - CENTER_X)
                    label = f"ENEMY {conf:.2f}"
                    if offset < closest_offset:
                        closest_offset = offset
                        closest_detection = (x1, y1, x2, y2, center_x)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if detections_items is not None and len(detections_items):
                detections_items[:, :4] = scale_boxes(img_letterboxed.shape[:2], detections_items[:, :4], img.shape).round()
                for *xyxy, conf, cls in detections_items:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls.item())
                    label = f"ITEM {class_id} {conf:.2f}"
                    color = (0, 255, 0) if class_id == 3 else (0, 255, 255) if class_id == 4 else (255, 255, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
        cv2.imshow("Freedoom Assist HUD", img)
        if cv2.waitKey(1) == 27:
            break

    game.make_action(actions)

    elapsed = time.time() - last_time
    time.sleep(max(0, FRAME_DURATION - elapsed))
    last_time = time.time()

game.close()
cv2.destroyAllWindows()
