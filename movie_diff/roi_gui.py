from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tkinter import (
    Tk,
    Canvas,
    Button,
    Frame,
    filedialog,
    messagebox,
    Label,
    Entry,
    Listbox,
    SINGLE,
)

from .models import ROI, ROISet
from .video import probe, read_first_frame
from .analyzer import analyze


@dataclass
class RectDrawing:
    start: Tuple[int, int] | None = None
    current: Tuple[int, int] | None = None
    temp_id: Optional[int] = None


class ROIToolApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("ROI Change-Rate Analyzer")

        # state
        self.video_path: Optional[str] = None
        self.first_frame_bgr: Optional[np.ndarray] = None
        self.display_image = None  # Tk photo image
        self.scale = 1.0
        self.offset = (0, 0)
        self.rois: List[ROI] = []

        # UI
        self.canvas = Canvas(self.root, width=960, height=540, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=6, pady=6)

        self.btn_open = Button(self.root, text="動画を開く", command=self.open_video)
        self.btn_open.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        self.btn_save = Button(self.root, text="ROI保存", command=self.save_rois, state="disabled")
        self.btn_save.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        self.btn_load = Button(self.root, text="ROI読み込み", command=self.load_rois)
        self.btn_load.grid(row=1, column=2, sticky="ew", padx=4, pady=2)

        self.btn_analyze = Button(self.root, text="解析してCSV出力", command=self.run_analysis, state="disabled")
        self.btn_analyze.grid(row=1, column=3, sticky="ew", padx=4, pady=2)

        Label(self.root, text="選択ROI").grid(row=2, column=0, sticky="w", padx=6)
        self.listbox = Listbox(self.root, selectmode=SINGLE, height=6)
        self.listbox.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=6, pady=2)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.refresh_canvas())

        Label(self.root, text="名前").grid(row=4, column=0, sticky="e", padx=6)
        self.entry_name = Entry(self.root)
        self.entry_name.grid(row=4, column=1, sticky="ew", padx=6)
        btn_set_name = Button(self.root, text="名前更新", command=self.update_selected_name)
        btn_set_name.grid(row=4, column=2, sticky="ew", padx=6)

        btn_delete = Button(self.root, text="選択削除", command=self.delete_selected)
        btn_delete.grid(row=3, column=2, sticky="ew", padx=6)

        Label(self.root, text="間引き(stride)").grid(row=5, column=0, sticky="e", padx=6)
        self.entry_stride = Entry(self.root)
        self.entry_stride.insert(0, "1")
        self.entry_stride.grid(row=5, column=1, sticky="ew", padx=6)

        # drawing handlers
        self.drawing = RectDrawing()
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # allow grid expand
        for r in range(6):
            self.root.grid_rowconfigure(r, weight=0)
        self.root.grid_rowconfigure(3, weight=1)
        for c in range(4):
            self.root.grid_columnconfigure(c, weight=1)

    def run(self):
        self.root.mainloop()

    # -------------- file ops --------------
    def open_video(self):
        path = filedialog.askopenfilename(title="動画を選択", filetypes=[("Video", "*.mp4;*.mov;*.avi;*.mkv;*.m4v;*.webm"), ("All", "*.*")])
        if not path:
            return
        try:
            frame = read_first_frame(path)
        except Exception as e:
            messagebox.showerror("Error", f"動画読み込みに失敗: {e}")
            return
        self.video_path = path
        self.first_frame_bgr = frame
        self.rois = []
        self.listbox.delete(0, "end")
        self.refresh_canvas()
        self.btn_save.config(state="normal")
        self.btn_analyze.config(state="normal")

    def save_rois(self):
        if not self.rois:
            messagebox.showinfo("Info", "ROIがありません")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        meta = probe(self.video_path) if self.video_path else None
        data = ROISet(
            video_path=self.video_path,
            frame_width=meta.width if meta else None,
            frame_height=meta.height if meta else None,
            rois=self.rois,
        ).dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("保存完了", f"ROIを保存しました\n{path}")

    def load_rois(self):
        path = filedialog.askopenfilename(title="ROI JSONを選択", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rs = ROISet(**data)
        except Exception as e:
            messagebox.showerror("Error", f"ROI読み込みに失敗: {e}")
            return
        self.rois = list(rs.rois)
        self.listbox.delete(0, "end")
        for i, r in enumerate(self.rois, start=1):
            self.listbox.insert("end", r.name or f"roi_{i}")
        self.refresh_canvas()

    # -------------- drawing --------------
    def image_to_canvas(self, x: int, y: int) -> Tuple[int, int]:
        ox, oy = self.offset
        return int(x * self.scale + ox), int(y * self.scale + oy)

    def canvas_to_image(self, x: int, y: int) -> Tuple[int, int]:
        ox, oy = self.offset
        return int((x - ox) / self.scale), int((y - oy) / self.scale)

    def refresh_canvas(self):
        self.canvas.delete("all")
        if self.first_frame_bgr is None:
            return
        # prepare image
        frame = self.first_frame_bgr
        h, w = frame.shape[:2]
        cw = int(self.canvas["width"])  # type: ignore
        ch = int(self.canvas["height"])  # type: ignore
        scale = min(cw / w, ch / h)
        self.scale = scale
        new_w, new_h = int(w * scale), int(h * scale)
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        self.offset = (ox, oy)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h))
        try:
            from PIL import Image, ImageTk  # type: ignore
            pil = Image.fromarray(resized)
            self.display_image = ImageTk.PhotoImage(pil)
        except Exception:
            # fallback using Tk's PhotoImage via PPM encoding
            import io
            from tkinter import PhotoImage

            success, buf = cv2.imencode(".ppm", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            if not success:
                raise RuntimeError("Failed to encode preview image")
            data = io.BytesIO(buf.tobytes()).getvalue()
            self.display_image = PhotoImage(data=data)

        self.canvas.create_image(ox, oy, anchor="nw", image=self.display_image)

        # draw existing ROIs
        sel = self.get_selected_index()
        for i, r in enumerate(self.rois):
            x1, y1 = self.image_to_canvas(r.x, r.y)
            x2, y2 = self.image_to_canvas(r.x + r.width, r.y + r.height)
            color = "#00FF00" if i == sel else "#FFCC00"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            name = r.name or f"roi_{i+1}"
            self.canvas.create_text(x1 + 4, y1 + 10, text=name, anchor="nw", fill=color)

        # draw temp rectangle
        if self.drawing.start and self.drawing.current:
            sx, sy = self.drawing.start
            cx, cy = self.drawing.current
            self.canvas.create_rectangle(sx, sy, cx, cy, outline="#00AAFF", width=2, dash=(4, 2))

    def on_mouse_down(self, event):
        if self.first_frame_bgr is None:
            return
        self.drawing.start = (event.x, event.y)
        self.drawing.current = (event.x, event.y)
        self.refresh_canvas()

    def on_mouse_drag(self, event):
        if self.drawing.start is None:
            return
        self.drawing.current = (event.x, event.y)
        self.refresh_canvas()

    def on_mouse_up(self, event):
        if self.drawing.start is None or self.first_frame_bgr is None:
            return
        sx, sy = self.drawing.start
        ex, ey = event.x, event.y
        self.drawing.start = None
        self.drawing.current = None
        x1, y1 = self.canvas_to_image(min(sx, ex), min(sy, ey))
        x2, y2 = self.canvas_to_image(max(sx, ex), max(sy, ey))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        h_img, w_img = self.first_frame_bgr.shape[:2]
        # clamp
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        if w <= 1 or h <= 1:
            self.refresh_canvas()
            return
        roi = ROI(x=int(x1), y=int(y1), width=int(w), height=int(h), name=None)
        self.rois.append(roi)
        self.listbox.insert("end", roi.name or f"roi_{len(self.rois)}")
        self.refresh_canvas()

    def get_selected_index(self) -> Optional[int]:
        sel = self.listbox.curselection()
        return int(sel[0]) if sel else None

    def delete_selected(self):
        idx = self.get_selected_index()
        if idx is None:
            return
        self.rois.pop(idx)
        self.listbox.delete(idx)
        self.refresh_canvas()

    def update_selected_name(self):
        idx = self.get_selected_index()
        if idx is None:
            return
        name = self.entry_name.get().strip()
        self.rois[idx].name = name or None
        self.listbox.delete(idx)
        self.listbox.insert(idx, name or f"roi_{idx+1}")
        self.refresh_canvas()

    # -------------- analysis --------------
    def run_analysis(self):
        if not self.video_path:
            messagebox.showinfo("Info", "動画を選択してください")
            return
        if not self.rois:
            messagebox.showinfo("Info", "ROIを指定してください")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out:
            return
        try:
            stride = int(self.entry_stride.get()) if self.entry_stride.get().strip() else 1
            if stride < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "strideは1以上の整数で指定してください")
            return

        # run in thread to keep UI responsive
        self.btn_analyze.config(state="disabled")
        self.btn_open.config(state="disabled")
        t = threading.Thread(target=self._analyze_thread, args=(self.video_path, list(self.rois), stride, out), daemon=True)
        t.start()

    def _analyze_thread(self, path: str, rois: List[ROI], stride: int, out_csv: str):
        try:
            analyze(path, rois, stride=stride, output_csv=out_csv, show_progress=True)
            self.root.after(0, lambda: messagebox.showinfo("完了", f"CSVを書き出しました\n{out_csv}"))
        except Exception as e:
            logger.exception(e)
            self.root.after(0, lambda: messagebox.showerror("Error", f"解析中にエラー: {e}"))
        finally:
            self.root.after(0, lambda: (self.btn_analyze.config(state="normal"), self.btn_open.config(state="normal")))
