import os
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")  # for headless image saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DATASET = "/Users/muhammadsaad/Desktop/HWU/YR4/Dissertation/Data Set/mathwriting-2024"
OUTPUT_ROOT = os.path.join(ROOT_DATASET, "processed")

# The dataset splits to process
SPLITS = ["train", "valid", "test"]

# Create processed folder if missing
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# XML namespace
ns = {'ink': 'http://www.w3.org/2003/InkML'}


# === PARSING FUNCTION ===
def parse_inkml(file_path):
    """Extract strokes and LaTeX label from InkML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"⚠️ Skipping unreadable file: {file_path} ({e})")
        return [], None

    # Extract all stroke traces
    traces = []
    for trace in root.findall('ink:trace', ns):
        coords = trace.text.strip().split(',')
        stroke = []
        for c in coords:
            xy = c.strip().split(' ')
            if len(xy) >= 2:
                try:
                    stroke.append((float(xy[0]), float(xy[1])))
                except ValueError:
                    continue
        if stroke:
            traces.append(np.array(stroke))

    # Extract label (normalizedLabel preferred)
    label = None
    for ann in root.findall('ink:annotation', ns):
        ann_type = ann.get('type', '').lower()
        if ann_type in ['normalizedlabel', 'label', 'truth', 'latex']:
            label = ann.text
            break

    return traces, label


# === NORMALIZATION ===
def normalize_strokes(traces, size=256, margin=10):
    """Normalize strokes to fit into a fixed square image."""
    if not traces:
        return []
    all_points = np.concatenate(traces)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    scale = (size - 2 * margin) / max(max_x - min_x, max_y - min_y)
    norm_traces = []
    for stroke in traces:
        norm = (stroke - [min_x, min_y]) * scale + margin
        norm_traces.append(norm)
    return norm_traces


# === RENDER FUNCTION ===
def render_strokes(traces, size=256):
    """Render normalized strokes to grayscale numpy image."""
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.axis('off')
    ax.invert_yaxis()
    for stroke in traces:
        ax.plot(stroke[:, 0], stroke[:, 1], 'k-', linewidth=2)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())
    img = np.mean(buf[:, :, :3], axis=2)  # RGBA → grayscale
    plt.close(fig)
    return img


# === PROCESS ONE SPLIT ===
def process_split(split_name):
    input_dir = os.path.join(ROOT_DATASET, split_name)
    output_img_dir = os.path.join(OUTPUT_ROOT, f"{split_name}_images")
    output_csv = os.path.join(OUTPUT_ROOT, f"{split_name}_labels.csv")

    os.makedirs(output_img_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".inkml")]
    print(f"\n📂 Processing split: {split_name} ({len(files)} files)")

    data = []
    for i, fname in enumerate(tqdm(files, desc=f"{split_name}")):
        path = os.path.join(input_dir, fname)
        traces, label = parse_inkml(path)
        if not traces or not label:
            continue

        norm_traces = normalize_strokes(traces)
        img = render_strokes(norm_traces)
        img_filename = f"{i:06d}.png"
        img_path = os.path.join(output_img_dir, img_filename)
        plt.imsave(img_path, img, cmap="gray")
        data.append({"filename": img_filename, "label": label})

    # Save CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"✅ {split_name.capitalize()} set saved: {len(data)} samples")
        print(f"   Images → {output_img_dir}")
        print(f"   Labels → {output_csv}")
    else:
        print(f"⚠️ No valid InkML files processed for {split_name}.")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)

    print("\n🎯 All dataset splits processed successfully!")