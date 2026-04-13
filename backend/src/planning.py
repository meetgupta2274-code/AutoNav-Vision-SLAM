import cv2
import numpy as np
from collections import defaultdict
import heapq
import math

class Planner:
    def __init__(self, nav_rows=30, nav_cols=30, grid_rows=3, grid_cols=3):
        self.nav_rows = nav_rows
        self.nav_cols = nav_cols
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Recommendations weights
        self.W_CONF = 0.6
        self.W_PATH = 0.3
        self.W_DIST = 0.1

    def zone_label(self, r, c):
        return f"{chr(65 + r)}{c + 1}"

    def pixel_to_cell(self, x, y, w, h, rows, cols):
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        col = int((x / w) * cols)
        row = int((y / h) * rows)
        return min(row, rows - 1), min(col, cols - 1)

    def cell_center_pixel(self, row, col, w, h, rows, cols):
        x0 = col * (w / cols)
        y0 = row * (h / rows)
        x1 = (col + 1) * (w / cols)
        y1 = (row + 1) * (h / rows)
        return int((x0 + x1) / 2), int((y0 + y1) / 2)

    def build_occupancy(self, h, w, obstacle_mask=None):
        occ = np.zeros((self.nav_rows, self.nav_cols), dtype=np.uint8)
        if obstacle_mask is None:
            return occ
        cell_h = h / self.nav_rows
        cell_w = w / self.nav_cols
        for r in range(self.nav_rows):
            for c in range(self.nav_cols):
                y0 = int(r * cell_h)
                y1 = int((r + 1) * cell_h)
                x0 = int(c * cell_w)
                x1 = int((c + 1) * cell_w)
                patch = obstacle_mask[y0:y1, x0:x1]
                if patch.size > 0 and np.any(patch > 0):
                    occ[r, c] = 1
        return occ

    def neighbors_8(self, row, col):
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            rr, cc = row + dr, col + dc
            if 0 <= rr < self.nav_rows and 0 <= cc < self.nav_cols:
                yield rr, cc

    def heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def astar(self, occ, start, goal):
        sr, sc = start
        gr, gc = goal
        if occ[sr, sc] == 1 or occ[gr, gc] == 1:
            return None
        g = {start: 0.0}
        f = {start: self.heuristic(start, goal)}
        came = {}
        openh = [(f[start], start)]
        in_open = {start}
        
        while openh:
            _, cur = heapq.heappop(openh)
            in_open.discard(cur)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path
            cr, cc = cur
            for nr, nc in self.neighbors_8(cr, cc):
                if occ[nr, nc] == 1:
                    continue
                step = 1.4142 if (nr != cr and nc != cc) else 1.0
                tentative = g[cur] + step
                if tentative < g.get((nr, nc), float('inf')):
                    came[(nr, nc)] = cur
                    g[(nr, nc)] = tentative
                    ff = tentative + self.heuristic((nr, nc), goal)
                    f[(nr, nc)] = ff
                    if (nr, nc) not in in_open:
                        heapq.heappush(openh, (ff, (nr, nc)))
                        in_open.add((nr, nc))
        return None

    def normalize(self, x, xmin, xmax):
        if xmax <= xmin:
            return 0.0
        return (x - xmin) / (xmax - xmin)

    def compute_recommendations(self, persons, start_px, occ, w, h):
        results = []
        if not persons:
            return results
            
        sx, sy = start_px
        sr, sc = self.pixel_to_cell(sx, sy, w, h, self.nav_rows, self.nav_cols)
        
        for p in persons:
            cx, cy = p["center"]
            tr, tc = self.pixel_to_cell(cx, cy, w, h, self.nav_rows, self.nav_cols)
            path = self.astar(occ, (sr, sc), (tr, tc))
            path_len = len(path) if path is not None else None
            dist_px = math.hypot(cx - sx, cy - sy)
            results.append({
                "person": p, 
                "target_cell": (tr, tc), 
                "path": path, 
                "path_len": path_len, 
                "dist_px": dist_px
            })
            
        confs = [r["person"]["conf"] for r in results] or [0.0]
        dists = [r["dist_px"] for r in results] or [1.0]
        paths = [r["path_len"] for r in results if r["path_len"] is not None]
        
        cmin, cmax = min(confs), max(confs)
        dmin, dmax = min(dists), max(dists)
        pmin, pmax = (min(paths), max(paths)) if paths else (1.0, 1.0)
        
        for r in results:
            conf_norm = self.normalize(r["person"]["conf"], cmin, cmax)
            path_norm = 1.0 - self.normalize(r["path_len"], pmin, pmax) if r["path_len"] is not None else 0.0
            dist_norm = 1.0 - self.normalize(r["dist_px"], dmin, dmax)
            r["score"] = self.W_CONF * conf_norm + self.W_PATH * path_norm + self.W_DIST * dist_norm
            
            reason = f"conf={r['person']['conf']:.2f}"
            reason += f", path={r['path_len']} cells" if r["path_len"] is not None else ", path=blocked"
            reason += f", dist={int(r['dist_px'])} px"
            r["reason"] = reason
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
