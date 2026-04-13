import cv2
import numpy as np
from .planning import Planner

class Visualizer:
    def __init__(self, planner: Planner):
        self.planner = planner
        self.DRAW_SECTOR_LABELS = True
        self.DRAW_ALL_PERSONS = True
        self.DRAW_PATH = True

    def draw_grid(self, img, rows, cols, color=(120, 120, 120), thickness=1, alpha=0.25):
        h, w = img.shape[:2]
        overlay = img.copy()
        for c in range(1, cols):
            x = int(c * w / cols)
            cv2.line(overlay, (x, 0), (x, h), color, thickness)
        for r in range(1, rows):
            y = int(r * h / rows)
            cv2.line(overlay, (0, y), (w, y), color, thickness)
        img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def draw_grid_with_labels(self, img, rows, cols, color=(120, 120, 120), thickness=1, alpha=0.25, label_alpha=0.6):
        self.draw_grid(img, rows, cols, color, thickness, alpha)
        h, w = img.shape[:2]
        label_overlay = img.copy()
        for r in range(rows):
            for c in range(cols):
                zl = self.planner.zone_label(r, c)
                x0 = int(c * w / cols)
                y0 = int(r * h / rows)
                (tw, th), _ = cv2.getTextSize(zl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(label_overlay, (x0 + 6, y0 + 6), (x0 + 10 + tw, y0 + 12 + th), (0, 0, 0), -1)
                cv2.putText(label_overlay, zl, (x0 + 10, y0 + 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        img[:] = cv2.addWeighted(label_overlay, label_alpha, img, 1 - label_alpha, 0)

    def draw_path(self, img, path, rows, cols, color=(0, 0, 255), thickness=3):
        if not path or len(path) < 2:
            return
        h, w = img.shape[:2]
        pts = []
        for (r, c) in path:
            x, y = self.planner.cell_center_pixel(r, c, w, h, rows, cols)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], color, thickness, cv2.LINE_AA)

    def overlay_sector_coords(self, frame, persons):
        h, w = frame.shape[:2]
        self.draw_grid_with_labels(frame, self.planner.grid_rows, self.planner.grid_cols)
        for p in persons:
            if "sector" not in p:
                cx, cy = p["center"]
                r, c = self.planner.pixel_to_cell(cx, cy, w, h, self.planner.grid_rows, self.planner.grid_cols)
                p["sector"] = self.planner.zone_label(r, c)

            x1, y1, x2, y2 = p["box"]
            cx, cy = p["center"]
            tid = f"ID:{p['id']} | " if p.get("id") is not None else ""
            tag = f"{tid}{p['sector']} | conf={p['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            tx, ty = x1, max(20, y1 - 8)
            cv2.rectangle(frame, (tx - 2, ty - th - 6), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
            cv2.putText(frame, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def annotate_frame(self, frame, persons, recs, start_px, occ):
        self.overlay_sector_coords(frame, persons)
        h, w = frame.shape[:2]
        self.draw_grid(frame, self.planner.nav_rows, self.planner.nav_cols)
        
        if recs:
            best = recs[0]
            sr, sc = self.planner.pixel_to_cell(start_px[0], start_px[1], w, h, self.planner.nav_rows, self.planner.nav_cols)
            gr, gc = best["target_cell"]
            sx, sy = self.planner.cell_center_pixel(sr, sc, w, h, self.planner.nav_rows, self.planner.nav_cols)
            gx, gy = self.planner.cell_center_pixel(gr, gc, w, h, self.planner.nav_rows, self.planner.nav_cols)
            path = best["path"] or self.planner.astar(occ, (sr, sc), (gr, gc))
            
            cv2.circle(frame, (sx, sy), 6, (0, 255, 0), -1)
            cv2.putText(frame, "START", (sx + 8, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (gx, gy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"GOAL {best['person'].get('sector','')}", (gx + 8, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if self.DRAW_PATH and path and len(path) > 1:
                self.draw_path(frame, path, self.planner.nav_rows, self.planner.nav_cols, color=(0, 0, 255), thickness=3)
                hud = f"ETA cells: {len(path)} | score={best['score']:.2f}"
                cv2.putText(frame, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No navigable path", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        return frame
