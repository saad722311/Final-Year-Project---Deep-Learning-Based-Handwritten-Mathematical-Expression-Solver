# src/ui/draw_symbol_canvas.py

import tkinter as tk
from PIL import Image, ImageDraw
import subprocess
import os
import time

OUT_DIR = "results/ui_samples"
OUT_IMG = os.path.join(OUT_DIR, "canvas.png")

SYMBOL_CMD = [
    "python3",
    "-m",
    "src.symbol_recognition.predict_symbol",  # <-- adjust if needed
    "--ckpt",
    "results/symbol_model/best.pt",            # <-- adjust if needed
    "--img",
    OUT_IMG,
]

os.makedirs(OUT_DIR, exist_ok=True)


class DrawCanvas:
    def __init__(self, root):
        self.root = root
        root.title("Mini Symbol Canvas")

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack(padx=10, pady=10)

        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Predict Symbol", command=self.predict).pack(side="left", padx=5)

        self.output = tk.Label(root, text="Draw a symbol and click Predict", font=("Arial", 12))
        self.output.pack(pady=8)

    def paint(self, event):
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=6, fill="black", capstyle=tk.ROUND
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="black", width=6
            )
        self.last_x = event.x
        self.last_y = event.y

    def reset(self, _):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self.output.config(text="Canvas cleared")

    def predict(self):
        self.image.save(OUT_IMG)
        self.output.config(text="Predicting...")

        try:
            result = subprocess.check_output(
                SYMBOL_CMD,
                stderr=subprocess.STDOUT,
                timeout=10
            ).decode("utf-8")

            self.output.config(text=result.strip())

        except Exception as e:
            self.output.config(text=f"Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    DrawCanvas(root)
    root.mainloop()