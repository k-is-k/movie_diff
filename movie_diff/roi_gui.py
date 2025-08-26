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
    Checkbutton,
    BooleanVar,
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
        self.analysis_df = None  # pandas DataFrame with analysis result
        self.graph_axis = None  # (gx0, gy0, gw, gh, x_min, x_max)
        self.series_colors = [
            "#ff4d4f",
            "#40a9ff",
            "#73d13d",
            "#faad14",
            "#9254de",
            "#13c2c2",
            "#eb2f96",
            "#a0d911",
        ]
        self.player_window = None  # type: ignore
        self._graph_prev_xview = None

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

        # Graph area (analysis result)
        Label(self.root, text="解析結果グラフ").grid(row=6, column=0, sticky="w", padx=6)
        # Toggle: per-ROI scaling
        self.var_split_scale = BooleanVar(value=True)
        self.chk_split = Checkbutton(
            self.root,
            text="ROIごとスケール",
            variable=self.var_split_scale,
            command=self.render_graph,
        )
        self.chk_split.grid(row=6, column=1, sticky="w")
        self.graph_canvas = Canvas(self.root, width=960, height=240, bg="white")
        self.graph_canvas.grid(row=7, column=0, columnspan=4, padx=6, pady=6, sticky="ew")
        # horizontal scrollbar for long videos
        from tkinter import Scrollbar
        self.graph_scroll = Scrollbar(self.root, orient="horizontal", command=self.graph_canvas.xview)
        self.graph_canvas.configure(xscrollcommand=self.graph_scroll.set)
        self.graph_scroll.grid(row=8, column=0, columnspan=4, padx=6, pady=(0, 6), sticky="ew")
        self.graph_canvas.bind("<Button-1>", self.on_graph_click)

    def run(self):
        self.root.mainloop()

    # -------------- file ops --------------
    def open_video(self):
        path = filedialog.askopenfilename(title="動画を選択", filetypes=[("Video", "*.mp4;*.mov;*.avi;*.mkv;*.m4v;*.webm;*.MTS"), ("All", "*.*")])
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
            df = analyze(path, rois, stride=stride, output_csv=out_csv, show_progress=True)
            def on_done():
                self.analysis_df = df
                self.render_graph()
                messagebox.showinfo("完了", f"CSVを書き出し、グラフを表示しました\n{out_csv}\nグラフをクリックすると該当位置から再生します")
            self.root.after(0, on_done)
        except Exception as e:
            logger.exception(e)
            self.root.after(0, lambda: messagebox.showerror("Error", f"解析中にエラー: {e}"))
        finally:
            self.root.after(0, lambda: (self.btn_analyze.config(state="normal"), self.btn_open.config(state="normal")))

    # -------------- graph rendering --------------
    def render_graph(self):
        c = self.graph_canvas
        c.delete("all")
        df = self.analysis_df
        if df is None or len(df) == 0:
            c.create_text(10, 10, anchor="nw", text="解析結果なし", fill="#888")
            self.graph_axis = None
            return
        try:
            import pandas as pd  # noqa: F401
        except Exception:
            pass

        # Layout
        w = int(c["width"])  # type: ignore
        h = int(c["height"])  # type: ignore
        margin_l, margin_r, margin_t, margin_b = 50, 10, 10, 30
        gx0, gy0 = margin_l, margin_t
        gw_view, gh_total = max(10, w - margin_l - margin_r), max(10, h - margin_t - margin_b)

        # Determine columns (series)
        cols = [col for col in df.columns if col not in ("frame_index", "timestamp_sec")]
        if not cols:
            c.create_text(10, 10, anchor="nw", text="系列がありません", fill="#888")
            self.graph_axis = None
            return

        # X range: frame_index
        try:
            x_values = df["frame_index"].values.tolist()
            x_min = min(x_values)
            x_max = max(x_values)
        except Exception:
            x_values = list(range(len(df)))
            x_min = 0
            x_max = max(1, len(df) - 1)

        # Determine world width and x-scale: ensure at least 1px per frame for scrolling
        x_span = max(1, int(x_max - x_min))
        scale_x = max(gw_view / x_span, 1.0)
        gw_world = x_span * scale_x

        def x_to_px(xv: float) -> float:
            if x_max == x_min:
                return gx0
            return gx0 + (xv - x_min) * scale_x

        # Draw either shared-scale or per-ROI panels
        split = bool(self.var_split_scale.get()) if hasattr(self, "var_split_scale") else False
        max_points = 4000
        n = len(df)
        step = max(1, n // max_points)

        if not split:
            # Shared scale: 0..1
            y_min, y_max = 0.0, 1.0
            def y_to_px(yv: float) -> float:
                return gy0 + gh_total - (yv - y_min) / (y_max - y_min) * gh_total

            # Background
            c.create_rectangle(gx0, gy0, gx0 + gw_world, gy0 + gh_total, outline="#ddd", fill="#fafafa")

            # Grid lines and ticks
            for i in range(0, 6):
                yy = gy0 + gh_total * i / 5
                val = 1 - i / 5
                c.create_line(gx0, yy, gx0 + gw_world, yy, fill="#eee")
                c.create_text(gx0 - 8, yy, text=f"{val:.1f}", anchor="e", fill="#999")

            # Legend
            legend_x, legend_y = gx0 + 8, gy0 + 8
            for idx, name in enumerate(cols):
                color = self.series_colors[idx % len(self.series_colors)]
                c.create_rectangle(legend_x, legend_y + idx * 16, legend_x + 12, legend_y + 12 + idx * 16, fill=color, outline=color)
                c.create_text(legend_x + 16, legend_y + 6 + idx * 16, text=name, anchor="w", fill="#555")

            # Draw all series
            for idx, name in enumerate(cols):
                color = self.series_colors[idx % len(self.series_colors)]
                pts = []
                for i in range(0, n, step):
                    xv = x_values[i]
                    try:
                        yv = float(df.iloc[i][name])
                    except Exception:
                        continue
                    px = x_to_px(xv)
                    py = y_to_px(max(0.0, min(1.0, yv)))
                    pts.extend([px, py])
                if len(pts) >= 4:
                    c.create_line(*pts, fill=color, width=2)

            # X labels
            for xv in (x_min, (x_min + x_max) / 2, x_max):
                xx = x_to_px(xv)
                c.create_line(xx, gy0 + gh_total, xx, gy0 + gh_total + 5, fill="#ccc")
                c.create_text(xx, gy0 + gh_total + 8, text=f"{int(round(xv))}", anchor="n", fill="#999")

            self.graph_axis = (gx0, gy0, gw_world, gh_total, x_min, x_max)
        else:
            # Per-ROI panels with individual y-scales
            num = max(1, len(cols))
            vgap = 8
            panel_h = max(10, int((gh_total - vgap * (num - 1)) / num))
            for idx, name in enumerate(cols):
                color = self.series_colors[idx % len(self.series_colors)]
                py0 = gy0 + idx * (panel_h + vgap)
                # Compute y-range for this series
                series_vals = []
                for i in range(0, n, step):
                    try:
                        series_vals.append(float(df.iloc[i][name]))
                    except Exception:
                        continue
                if series_vals:
                    s_min = min(series_vals)
                    s_max = max(series_vals)
                else:
                    s_min, s_max = 0.0, 1.0
                if abs(s_max - s_min) < 1e-6:
                    pad = 0.05 if s_max == 0 else s_max * 0.05
                    s_min -= pad
                    s_max += pad

                def y_to_px_local(yv: float, ymin=s_min, ymax=s_max, base_y=py0):
                    return base_y + panel_h - (yv - ymin) / (ymax - ymin) * panel_h

                # Panel background
                c.create_rectangle(gx0, py0, gx0 + gw_world, py0 + panel_h, outline="#ddd", fill="#fafafa")

                # Grid and ticks (min/mid/max)
                for j, val in enumerate([s_max, (s_min + s_max) / 2, s_min]):
                    yy = y_to_px_local(val)
                    c.create_line(gx0, yy, gx0 + gw_world, yy, fill="#eee")
                    c.create_text(gx0 - 8, yy, text=f"{val:.3f}", anchor="e", fill="#999")

                # Series label with color swatch
                c.create_rectangle(gx0 + 8, py0 + 6, gx0 + 20, py0 + 18, fill=color, outline=color)
                c.create_text(gx0 + 24, py0 + 12, text=name, anchor="w", fill="#555")

                # Draw the line for this series
                pts = []
                for i in range(0, n, step):
                    xv = x_values[i]
                    try:
                        yv = float(df.iloc[i][name])
                    except Exception:
                        continue
                    px = x_to_px(xv)
                    py = y_to_px_local(yv)
                    pts.extend([px, py])
                if len(pts) >= 4:
                    c.create_line(*pts, fill=color, width=2)

            # X labels at bottom only
            for xv in (x_min, (x_min + x_max) / 2, x_max):
                xx = x_to_px(xv)
                c.create_line(xx, gy0 + gh_total, xx, gy0 + gh_total + 5, fill="#ccc")
                c.create_text(xx, gy0 + gh_total + 8, text=f"{int(round(xv))}", anchor="n", fill="#999")

            self.graph_axis = (gx0, gy0, gw_world, gh_total, x_min, x_max)

        # Configure scrollregion to enable horizontal scrolling across world width
        total_w = margin_l + gw_world + margin_r
        total_h = h
        c.configure(scrollregion=(0, 0, total_w, total_h))
        # restore previous xview if available
        if self._graph_prev_xview is not None:
            try:
                c.xview_moveto(self._graph_prev_xview[0])
            except Exception:
                pass
        # store current xview
        try:
            self._graph_prev_xview = c.xview()
        except Exception:
            self._graph_prev_xview = None

    def on_graph_click(self, event):
        if not self.video_path or self.analysis_df is None or self.graph_axis is None:
            return
        # Snap click to the nearest analyzed (sampled) frame to avoid
        # apparent N× misalignment when stride > 1.
        gx0, gy0, gw, gh, x_min, x_max = self.graph_axis
        df = self.analysis_df
        if df is None or len(df) == 0:
            return
        # translate event.x (viewport coords) to canvas/world coords for correct mapping when scrolled
        xw = self.graph_canvas.canvasx(event.x)
        x = max(gx0, min(gx0 + gw, xw))
        ratio = 0.0 if gw == 0 else (x - gx0) / gw
        row_idx = int(round(ratio * (len(df) - 1)))
        try:
            frame_idx = int(df.iloc[row_idx]["frame_index"])  # exact sampled frame
        except Exception:
            # Fallback to linear mapping over frame range
            frame_idx = int(round(x_min + ratio * (x_max - x_min)))
        # open player window at this frame
        self.open_player(frame_idx)

    # -------------- simple video player --------------
    def open_player(self, start_frame: int = 0):
        if not self.video_path:
            return
        # reuse existing player if open to avoid heavy re-open
        pw = getattr(self, "player_window", None)
        if pw and getattr(pw, "alive", False):
            try:
                pw.jump_to(start_frame)
                return
            except Exception:
                pass
        self.player_window = PlayerWindow(self.root, self.video_path, start_frame=start_frame)


class PlayerWindow:
    def __init__(self, master, video_path: str, start_frame: int = 0) -> None:
        from tkinter import Toplevel, Scale, HORIZONTAL

        self.top = Toplevel(master)
        self.top.title("プレビュー再生")
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "動画を開けませんでした")
            self.top.destroy()
            return
        self.alive = True
        meta = probe(video_path)
        self.fps = meta.fps if meta and meta.fps else float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (meta.n_frames if meta and meta.n_frames else 0)
        self.playing = False
        self.current_frame = max(0, min(start_frame, max(0, self.n_frames - 1)))
        self.photo = None

        # UI widgets
        self.canvas = Canvas(self.top, width=640, height=360, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=6, pady=6)

        self.scale = Scale(self.top, from_=0, to=max(0, self.n_frames - 1), orient=HORIZONTAL, length=640, command=self.on_seek)
        self.scale.grid(row=1, column=0, columnspan=4, padx=6, sticky="ew")

        self.btn_play = Button(self.top, text="再生", command=self.toggle_play)
        self.btn_play.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        Button(self.top, text="<< -30f", command=lambda: self.jump(-30)).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        Button(self.top, text="-1f", command=lambda: self.jump(-1)).grid(row=2, column=2, sticky="ew", padx=4, pady=4)
        Button(self.top, text="+1f", command=lambda: self.jump(1)).grid(row=2, column=3, sticky="ew", padx=4, pady=4)

        self.top.protocol("WM_DELETE_WINDOW", self.on_close)

        # initial frame
        self.seek_to(self.current_frame)
        self.render_current_frame()

    def on_close(self):
        self.playing = False
        try:
            self.cap.release()
        except Exception:
            pass
        self.alive = False
        self.top.destroy()

    def on_seek(self, value):
        try:
            idx = int(float(value))
        except Exception:
            return
        self.current_frame = max(0, min(idx, max(0, self.n_frames - 1)))
        self.seek_to(self.current_frame)
        self.render_current_frame()

    def seek_to(self, frame_idx: int):
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        except Exception:
            pass

    def jump(self, delta: int):
        self.current_frame = max(0, min(self.current_frame + delta, max(0, self.n_frames - 1)))
        self.scale.set(self.current_frame)
        self.seek_to(self.current_frame)
        self.render_current_frame()

    def jump_to(self, frame_idx: int):
        # external control to jump directly to a frame (used from graph)
        self.playing = False
        try:
            self.btn_play.config(text="再生")
        except Exception:
            pass
        self.current_frame = max(0, min(int(frame_idx), max(0, self.n_frames - 1)))
        self.scale.set(self.current_frame)
        self.seek_to(self.current_frame)
        self.render_current_frame()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="停止" if self.playing else "再生")
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if not self.playing:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.btn_play.config(text="再生")
            return
        self.current_frame += 1
        self.scale.set(self.current_frame)
        self.render_frame(frame)
        delay = int(1000 / max(1, int(round(self.fps))))
        self.top.after(delay, self.play_loop)

    def render_current_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        self.render_frame(frame)

    def render_frame(self, frame):
        # fit into canvas while preserving aspect
        ch = int(self.canvas["height"])  # type: ignore
        cw = int(self.canvas["width"])  # type: ignore
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h))
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        try:
            from PIL import Image, ImageTk  # type: ignore
            pil = Image.fromarray(resized)
            self.photo = ImageTk.PhotoImage(pil)
        except Exception:
            import io
            from tkinter import PhotoImage
            success, buf = cv2.imencode(".ppm", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            if not success:
                return
            data = io.BytesIO(buf.tobytes()).getvalue()
            self.photo = PhotoImage(data=data)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
