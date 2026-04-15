import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from ultralytics import YOLO
import threading

# ================= CONFIG ================= #
MODEL_PATH = r"best.pt"
CLASS_NAMES = ["damaged_pill"]
CONF_THRESHOLD = 0.5
IMG_SIZE = 640


# ================= DETECTION ================= #
def detect_pills(image_path, model, conf=CONF_THRESHOLD):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(source=str(image_path),
                            conf=conf,
                            imgsz=IMG_SIZE,
                            verbose=False)

    boxes = results[0].boxes
    annotated = img_rgb.copy()

    details = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        cls = int(box.cls[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 30, 30), 2)

        label = f"{CLASS_NAMES[cls]} {confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        details.append({
            "id": i + 1,
            "conf": confidence,
            "box": (x1, y1, x2, y2)
        })

    return img_rgb, annotated, len(boxes), details


# ================= APP ================= #
class PillDetectorApp:

    def __init__(self, root):
        self.root = root
        self.model = None
        self.image_path = None
        self.annotated_img = None

        self.root.title("Pill Damage Detector")
        self.root.geometry("1150x720")
        self.root.configure(bg="#1e1e2e")

        self.setup_style()
        self.build_ui()
        self.load_model()

        # shortcuts
        self.root.bind("<Control-o>", lambda e: self.upload_image())
        self.root.bind("<Return>", lambda e: self.run_detection())

    # ================= STYLE ================= #
    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TLabel", font=("Segoe UI", 10))

    def hover(self, btn, enter, leave):
        btn.bind("<Enter>", lambda e: btn.config(bg=enter))
        btn.bind("<Leave>", lambda e: btn.config(bg=leave))

    # ================= UI ================= #
    def build_ui(self):

        # Title
        title = tk.Frame(self.root, bg="#181825", pady=12)
        title.pack(fill="x")

        tk.Label(title, text="💊 Pill Damage Detection System",
                 font=("Segoe UI", 18, "bold"),
                 fg="#cdd6f4", bg="#181825").pack()

        # Status
        self.status_var = tk.StringVar(value="🚀 Initializing AI Model...")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#313244", fg="#a6e3a1",
                 anchor="w", padx=10).pack(fill="x")

        # Layout
        content = tk.Frame(self.root, bg="#1e1e2e")
        content.pack(fill="both", expand=True, padx=12, pady=10)

        left = tk.Frame(content, bg="#1e1e2e", width=260)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        right = tk.Frame(content, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True)

        self.build_left(left)
        self.build_right(right)

    def build_left(self, parent):

        # Upload
        self.upload_btn = tk.Button(parent, text="📂 Upload Image",
                                    bg="#89b4fa", fg="#1e1e2e",
                                    relief="flat", pady=10,
                                    command=self.upload_image)
        self.upload_btn.pack(fill="x", pady=6)
        self.hover(self.upload_btn, "#74c7ec", "#89b4fa")

        # Detect
        self.detect_btn = tk.Button(parent, text="🔍 Detect",
                                    bg="#a6e3a1", fg="#1e1e2e",
                                    relief="flat", pady=10,
                                    state="disabled",
                                    command=self.run_detection)
        self.detect_btn.pack(fill="x", pady=6)
        self.hover(self.detect_btn, "#94e2d5", "#a6e3a1")

        # Save
        self.save_btn = tk.Button(parent, text="💾 Save Result",
                                  bg="#313244", fg="#cdd6f4",
                                  relief="flat",
                                  state="disabled",
                                  command=self.save_result)
        self.save_btn.pack(fill="x", pady=6)

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", pady=10)

        # Result Card
        card = tk.Frame(parent, bg="#313244",
                        highlightbackground="#45475a",
                        highlightthickness=1,
                        padx=10, pady=10)
        card.pack(fill="x", pady=10)

        tk.Label(card, text="Detected",
                 bg="#313244", fg="#6c7086").pack(anchor="w")

        self.count_label = tk.Label(card, text="—",
                                    font=("Segoe UI", 28, "bold"),
                                    bg="#313244", fg="#f38ba8")
        self.count_label.pack(anchor="w")

        # List
        self.listbox = tk.Listbox(parent,
                                  bg="#313244",
                                  fg="#cdd6f4")
        self.listbox.pack(fill="both", expand=True)

    def build_right(self, parent):

        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        tk.Label(parent, text="Original",
                 bg="#1e1e2e", fg="#6c7086").grid(row=0, column=0)

        tk.Label(parent, text="Result",
                 bg="#1e1e2e", fg="#6c7086").grid(row=0, column=1)

        self.canvas_orig = tk.Label(parent,
                                    bg="#313244",
                                    relief="ridge",
                                    bd=2)
        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.canvas_result = tk.Label(parent,
                                      bg="#313244",
                                      relief="ridge",
                                      bd=2)
        self.canvas_result.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    # ================= MODEL ================= #
    def load_model(self):
        def load():
            try:
                self.model = YOLO(MODEL_PATH)
                self.status_var.set("✅ Model Loaded")
                self.upload_btn.config(state="normal")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=load, daemon=True).start()

    # ================= HELPERS ================= #
    def resize_img(self, img, widget):
        self.root.update_idletasks()
        w = widget.winfo_width()
        h = widget.winfo_height()
        if w < 10 or h < 10:
            w, h = 400, 400

        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        new = cv2.resize(img, (int(iw * scale), int(ih * scale)))

        return ImageTk.PhotoImage(Image.fromarray(new))

    # ================= ACTIONS ================= #
    def upload_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.image_path = path
        img = np.array(Image.open(path).convert("RGB"))

        photo = self.resize_img(img, self.canvas_orig)
        self.canvas_orig.config(image=photo)
        self.canvas_orig.image = photo

        self.detect_btn.config(state="normal")
        self.status_var.set("📂 Image Loaded")

    def run_detection(self):
        if not self.image_path:
            return

        self.progress.start()
        self.status_var.set("🔍 Running Detection...")

        def detect():
            try:
                orig, annotated, count, details = detect_pills(
                    self.image_path, self.model,
                    CONF_THRESHOLD
                )

                self.annotated_img = annotated

                photo = self.resize_img(annotated, self.canvas_result)
                self.canvas_result.config(image=photo)
                self.canvas_result.image = photo

                self.count_label.config(text=str(count))

                if count == 0:
                    self.count_label.config(fg="#a6e3a1")
                else:
                    self.count_label.config(fg="#f38ba8")

                self.listbox.delete(0, tk.END)
                for d in details:
                    self.listbox.insert(tk.END,
                                        f"#{d['id']} conf={d['conf']:.2f}")

                self.status_var.set(f"✅ Done: {count} detected")
                self.save_btn.config(state="normal")

            except Exception as e:
                messagebox.showerror("Error", str(e))

            self.progress.stop()

        threading.Thread(target=detect, daemon=True).start()

    def save_result(self):
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            cv2.imwrite(path,
                        cv2.cvtColor(self.annotated_img,
                                     cv2.COLOR_RGB2BGR))
            self.status_var.set("💾 Saved successfully")


# ================= RUN ================= #
if __name__ == "__main__":
    root = tk.Tk()
    app = PillDetectorApp(root)
    root.mainloop()