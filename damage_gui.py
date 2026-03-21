import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from ultralytics import YOLO
import threading

# ============================================================
# CONFIGURATION — Update this path to your best.pt
# ============================================================
MODEL_PATH     = r"best.pt"
CLASS_NAMES    = ["damaged_pill"]
CONF_THRESHOLD = 0.25
IMG_SIZE       = 640

# ============================================================
# Core Detection Function
# ============================================================
def detect_pills(image_path, model, conf=CONF_THRESHOLD):
    """Run inference and return (original_img, annotated_img, count, details)."""
    img      = cv2.imread(str(image_path))
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results   = model.predict(source=str(image_path), conf=conf,
                               imgsz=IMG_SIZE, verbose=False)
    boxes     = results[0].boxes
    annotated = img_rgb.copy()

    details = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence      = float(box.conf[0])
        cls             = int(box.cls[0])

        # Red bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 30, 30), 2)

        # Label
        label = f"{CLASS_NAMES[cls]} {confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (220, 30, 30), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        details.append({"id": i + 1, "conf": confidence, "box": (x1, y1, x2, y2)})

    return img_rgb, annotated, len(boxes), details


# ============================================================
# GUI Application
# ============================================================
class PillDetectorApp:
    def __init__(self, root):
        self.root       = root
        self.model      = None
        self.image_path = None
        self.annotated_img = None

        self.root.title("Pill Damage Detector")
        self.root.geometry("1100x720")
        self.root.resizable(True, True)
        self.root.configure(bg="#1e1e2e")

        self._build_ui()
        self._load_model()

    # ----------------------------------------------------------
    # UI Layout
    # ----------------------------------------------------------
    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="#181825", pady=12)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="Pill Damage Detection System",
                 font=("Segoe UI", 18, "bold"),
                 fg="#cdd6f4", bg="#181825").pack()
        tk.Label(title_frame, text="Machine Vision Project  |  YOLOv8",
                 font=("Segoe UI", 10), fg="#6c7086", bg="#181825").pack()

        # Status bar
        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 10), fg="#a6e3a1",
                 bg="#313244", anchor="w", padx=12, pady=5).pack(fill="x")

        # Main content
        content = tk.Frame(self.root, bg="#1e1e2e")
        content.pack(fill="both", expand=True, padx=16, pady=10)

        # Left panel
        left = tk.Frame(content, bg="#1e1e2e", width=260)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        # Right panel
        right = tk.Frame(content, bg="#1e1e2e")
        right.pack(side="left", fill="both", expand=True)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        # Upload button
        tk.Label(parent, text="IMAGE INPUT", font=("Segoe UI", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", pady=(10, 4))

        self.upload_btn = tk.Button(
            parent, text="Upload Image",
            font=("Segoe UI", 11, "bold"),
            bg="#89b4fa", fg="#1e1e2e",
            activebackground="#74c7ec",
            relief="flat", cursor="hand2", pady=10,
            command=self.upload_image
        )
        self.upload_btn.pack(fill="x", pady=(0, 8))

        # Confidence slider
        tk.Label(parent, text="CONFIDENCE THRESHOLD",
                 font=("Segoe UI", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", pady=(10, 2))

        self.conf_var   = tk.DoubleVar(value=CONF_THRESHOLD)
        self.conf_label = tk.Label(parent, text=f"{CONF_THRESHOLD:.2f}",
                                   font=("Segoe UI", 11, "bold"),
                                   fg="#cdd6f4", bg="#1e1e2e")
        self.conf_label.pack(anchor="w")

        tk.Scale(parent, from_=0.05, to=0.95, resolution=0.05,
                 orient="horizontal", variable=self.conf_var,
                 bg="#1e1e2e", fg="#cdd6f4", highlightthickness=0,
                 troughcolor="#313244", activebackground="#89b4fa",
                 command=lambda v: self.conf_label.config(text=f"{float(v):.2f}")
                 ).pack(fill="x")

        # Detect button
        self.detect_btn = tk.Button(
            parent, text="Detect Damaged Pills",
            font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1", fg="#1e1e2e",
            activebackground="#94e2d5",
            relief="flat", cursor="hand2", pady=10,
            state="disabled", command=self.run_detection
        )
        self.detect_btn.pack(fill="x", pady=(14, 4))

        # Save button
        self.save_btn = tk.Button(
            parent, text="Save Result",
            font=("Segoe UI", 10),
            bg="#313244", fg="#cdd6f4",
            activebackground="#45475a",
            relief="flat", cursor="hand2", pady=8,
            state="disabled", command=self.save_result
        )
        self.save_btn.pack(fill="x", pady=(0, 16))

        # Results
        tk.Label(parent, text="DETECTION RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", pady=(10, 4))

        result_frame = tk.Frame(parent, bg="#313244", padx=12, pady=12)
        result_frame.pack(fill="x")

        self.count_var = tk.StringVar(value="—")
        tk.Label(result_frame, text="Damaged Pills Found",
                 font=("Segoe UI", 9), fg="#6c7086", bg="#313244").pack(anchor="w")
        tk.Label(result_frame, textvariable=self.count_var,
                 font=("Segoe UI", 28, "bold"),
                 fg="#f38ba8", bg="#313244").pack(anchor="w")

        # Detail list
        tk.Label(parent, text="DETECTIONS",
                 font=("Segoe UI", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", pady=(14, 4))

        list_frame = tk.Frame(parent, bg="#313244")
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.detail_list = tk.Listbox(
            list_frame, font=("Consolas", 9),
            bg="#313244", fg="#cdd6f4",
            selectbackground="#89b4fa",
            relief="flat", bd=0,
            yscrollcommand=scrollbar.set
        )
        self.detail_list.pack(fill="both", expand=True, padx=4, pady=4)
        scrollbar.config(command=self.detail_list.yview)

    def _build_right_panel(self, parent):
        tk.Label(parent, text="ORIGINAL IMAGE",
                 font=("Segoe UI", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").grid(row=0, column=0, sticky="w", padx=4)
        tk.Label(parent, text="DETECTED RESULT",
                 font=("Segoe UI", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").grid(row=0, column=1, sticky="w", padx=4)

        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        self.canvas_orig = tk.Label(parent, bg="#313244",
                                    text="Upload an image\nto get started",
                                    font=("Segoe UI", 12), fg="#6c7086")
        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        self.canvas_result = tk.Label(parent, bg="#313244",
                                      text="Detection result\nwill appear here",
                                      font=("Segoe UI", 12), fg="#6c7086")
        self.canvas_result.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

    # ----------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------
    def _load_model(self):
        def load():
            try:
                self.model = YOLO(MODEL_PATH)
                self.status_var.set("Model loaded — ready to detect!")
                self.upload_btn.config(state="normal")
            except Exception as e:
                self.status_var.set(f"Model load failed: {e}")
                messagebox.showerror("Model Error",
                    f"Could not load model from:\n{MODEL_PATH}\n\nError: {e}")
        threading.Thread(target=load, daemon=True).start()

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _resize_for_canvas(self, img_array, canvas_widget):
        self.root.update_idletasks()
        cw = canvas_widget.winfo_width()
        ch = canvas_widget.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 480, 480
        h, w   = img_array.shape[:2]
        scale  = min(cw / w, ch / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        resized = cv2.resize(img_array, (new_w, new_h))
        return ImageTk.PhotoImage(Image.fromarray(resized))

    # ----------------------------------------------------------
    # Actions
    # ----------------------------------------------------------
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Pill Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        self.image_path    = path
        self.annotated_img = None

        img   = np.array(Image.open(path).convert("RGB"))
        photo = self._resize_for_canvas(img, self.canvas_orig)
        self.canvas_orig.config(image=photo, text="")
        self.canvas_orig.image = photo

        self.canvas_result.config(image="",
                                   text="Click Detect\nto run detection",
                                   fg="#6c7086")
        self.canvas_result.image = None

        self.count_var.set("—")
        self.detail_list.delete(0, tk.END)
        self.detect_btn.config(state="normal")
        self.save_btn.config(state="disabled")
        self.status_var.set(f"Loaded: {Path(path).name}")

    def run_detection(self):
        if not self.image_path or not self.model:
            return

        self.detect_btn.config(state="disabled")
        self.status_var.set("Running detection...")

        def detect():
            try:
                conf = self.conf_var.get()
                orig, annotated, count, details = detect_pills(
                    self.image_path, self.model, conf
                )
                self.annotated_img = annotated

                photo = self._resize_for_canvas(annotated, self.canvas_result)
                self.canvas_result.config(image=photo, text="")
                self.canvas_result.image = photo

                self.count_var.set(str(count))

                self.detail_list.delete(0, tk.END)
                if count == 0:
                    self.detail_list.insert(tk.END, "  No damaged pills found")
                else:
                    for d in details:
                        x1, y1, x2, y2 = d["box"]
                        self.detail_list.insert(
                            tk.END,
                            f"  #{d['id']}  conf={d['conf']:.3f}  [{x1},{y1} -> {x2},{y2}]"
                        )

                msg = (f"Done — {count} damaged pill(s) detected"
                       if count > 0 else "Done — No damage found")
                self.status_var.set(msg)
                self.detect_btn.config(state="normal")
                self.save_btn.config(state="normal")

            except Exception as e:
                self.status_var.set(f"Detection failed: {e}")
                self.detect_btn.config(state="normal")
                messagebox.showerror("Detection Error", str(e))

        threading.Thread(target=detect, daemon=True).start()

    def save_result(self):
        if self.annotated_img is None:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"detected_{Path(self.image_path).stem}.jpg"
        )
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(self.annotated_img, cv2.COLOR_RGB2BGR))
            self.status_var.set(f"Saved to: {save_path}")
            messagebox.showinfo("Saved!", f"Result saved to:\n{save_path}")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = PillDetectorApp(root)
    root.mainloop()