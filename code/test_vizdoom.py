import time
import os
import cv2
import pandas as pd
import keyboard
from vizdoom import DoomGame, Mode, ScreenResolution

# === Configuración de guardado ===
SAVE_PATH = "dataset"
FRAME_PATH = os.path.join(SAVE_PATH, "frames")
CSV_PATH = os.path.join(SAVE_PATH, "metadata.csv")

os.makedirs(FRAME_PATH, exist_ok=True)
data = []
frame_id = 0
frame_counter = 0

# === Inicializar juego ===
game = DoomGame()
game.load_config("doom1.cfg")
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.set_screen_resolution(ScreenResolution.RES_1024X768)
game.init()
game.new_episode()

print("Controles activos. Presiona ESC para salir.")

while not game.is_episode_finished():
    actions = [0] * 8  # Asegúrate que coincida con tus botones en doom1.cfg

    # Asigna acciones con teclado
    if keyboard.is_pressed("space"): actions[0] = 1
    if keyboard.is_pressed("e"):     actions[1] = 1
    if keyboard.is_pressed("left"):  actions[2] = 1
    if keyboard.is_pressed("right"): actions[3] = 1
    if keyboard.is_pressed("d"):     actions[4] = 1
    if keyboard.is_pressed("a"):     actions[5] = 1
    if keyboard.is_pressed("w"):     actions[6] = 1
    if keyboard.is_pressed("s"):     actions[7] = 1

    # Ejecutar acción
    reward = game.make_action(actions)

    # Guardar imagen y datos cada 30 frames
    if frame_counter % 10 == 0:
        state = game.get_state()
        if state and state.screen_buffer is not None and state.screen_buffer.shape[0] == 3:
            
            frame_filename = f"{frame_id}.png"
            frame = state.screen_buffer.transpose(1, 2, 0)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

game.close()

# Guardar metadata al final
df = pd.DataFrame(data)
df.to_csv(CSV_PATH, index=False)
print(f"\n🎉 Datos guardados: {frame_id} imágenes y CSV con acciones.")
