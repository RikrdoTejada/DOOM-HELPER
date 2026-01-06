import os

#  Directory containing YOLO .txt annotation files
label_dir = "C:/Users/User/OneDrive/Documents/GitHub/VizDoomIA/dataset/labels/train"

#  Filter only .txt files (and skip classes.txt)
txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt') and f != 'classes.txt']

for file in txt_files:
    file_path = os.path.join(label_dir, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 🧹 Keep only lines that start with '0 '
    filtered_lines = [line for line in lines if line.strip().startswith('0 ')]

    with open(file_path, 'w') as f:
        f.writelines(filtered_lines)

    print(f" Cleaned: {file} - {len(filtered_lines)} tag(s) kept")
