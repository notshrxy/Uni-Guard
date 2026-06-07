import cv2
import numpy as np
import sys
import ctypes
import time
import random
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

screen_w = ctypes.windll.user32.GetSystemMetrics(0)
screen_h = ctypes.windll.user32.GetSystemMetrics(1)

# High-Fidelity Color Palette (BGR)
bg_color = (28, 25, 25)          
panel_bg = (34, 32, 32)          
border_color = (48, 45, 45)      
text_primary = (230, 230, 230)
text_secondary = (130, 130, 130)

blue_accent = (255, 160, 60)     
green_accent = (100, 200, 50)
orange_accent = (0, 150, 255)
red_accent = (50, 50, 255)
yellow_accent = (0, 200, 255)
purple_accent = (255, 50, 150)
grey_accent = (150, 150, 150)

# Global State
mouse_x, mouse_y = 0, 0
quit_app = False
target_tag = "purple"
event_log = [f"{datetime.now().strftime('%H:%M:%S')} - System Ready: Node 01 Online"]

# Graphs state
last_graph_update = time.time() - 30 
conf_data = []
prec_data = []


# ==========================================
# PERFECT ROUNDED RECTANGLE UTILITY
# ==========================================
def draw_rr(img, x, y, w, h, color, r, filled=True, thickness=1):
    x, y, w, h, r = int(x), int(y), int(w), int(h), int(r)
    if filled:
        cv2.circle(img, (x+r, y+r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x+w-r-1, y+r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x+r, y+h-r-1), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x+w-r-1, y+h-r-1), r, color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x+r, y), (x+w-r-1, y+h-1), color, -1)
        cv2.rectangle(img, (x, y+r), (x+w-1, y+h-r-1), color, -1)
    else:
        cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r-1, y+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r-1, y+h-r-1), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+h-r-1), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+r, y), (x+w-r-1, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+r, y+h-1), (x+w-r-1, y+h-1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y+r), (x, y+h-r-1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+w-1, y+r), (x+w-1, y+h-r-1), color, thickness, cv2.LINE_AA)

# ==========================================
# WINDOW ENGINE
# ==========================================
class Panel:
    def __init__(self, id, name, draw_fn, tw, th, active=True):
        self.id = id
        self.name = name
        self.draw_fn = draw_fn
        self.active = active
        self.tw = tw
        self.th = th
        self.rect = [screen_w//2, screen_h, tw, th]
        self.target_rect = [screen_w//2, screen_h, tw, th]
        
    def update(self):
        for i in range(4): 
            self.rect[i] += (self.target_rect[i] - self.rect[i]) * 0.15

    def render(self, canvas, global_frame):
        x, y, w, h = map(int, self.rect)
        if w < 10 or h < 10: return
        
        tw, th = self.target_rect[2], self.target_rect[3]
        if tw < 10 or th < 10: tw, th = w, h
            
        buffer = np.zeros((th, tw, 3), dtype=np.uint8)
        buffer[:] = bg_color 
        
        draw_rr(buffer, 0, 0, tw, th, panel_bg, 12, filled=True)
        draw_rr(buffer, 0, 0, tw, th, border_color, 12, filled=False, thickness=1)
        
        self.draw_fn(buffer, tw, th, global_frame)
        
        if tw != w or th != h:
            buffer = cv2.resize(buffer, (w, h))
            
        if x < 0 or y < 0 or x+w > screen_w or y+h > screen_h:
            return
        # Mask out bg_color to keep rounded corners clean when blitting
        mask = np.all(buffer == bg_color, axis=-1)
        canvas_roi = canvas[y:y+h, x:x+w]
        np.copyto(canvas_roi, buffer, where=~mask[..., None])

class DockIcon:
    def __init__(self, panel_id, name):
        self.panel_id = panel_id
        self.name = name
        self.base_w = 40
        self.target_w = 40
        self.w = 40
        self.x = 0
        self.y = 0
        
    def update(self, mx, my):
        # Hover Magnification
        cx = self.x + self.w / 2
        cy = self.y + self.w / 2
        dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
        if dist < 60:
            self.target_w = 55
        else:
            self.target_w = self.base_w
        self.w += (self.target_w - self.w) * 0.2
        
    def render(self, canvas, active):
        w = int(self.w)
        y_offset = int((self.base_w - w) / 2)
        dy = int(self.y) + y_offset
        dx = int(self.x) - int((w - self.base_w)/2)
        
        bg_col = (50, 40, 30) if active else panel_bg 
        
        draw_rr(canvas, dx, dy, w, w, bg_col, 10, filled=True)
        if active:
            draw_rr(canvas, dx, dy, w, w, blue_accent, 10, filled=False, thickness=2)
        else:
            draw_rr(canvas, dx, dy, w, w, border_color, 10, filled=False, thickness=1)
            
        letter = self.name[0].upper()
        fs = w / 70.0
        ts = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
        cv2.putText(canvas, letter, (dx + w//2 - ts[0]//2, dy + w//2 + ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, fs, text_primary, 2, cv2.LINE_AA)
        
        # Name below icon
        ns = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(canvas, self.name.upper(), (dx + w//2 - ns[0]//2, dy + w + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)


panels = {}
dock_icons = []

# ==========================================
# MODULE RENDERING LOGIC
# ==========================================
def draw_camera(buffer, w, h, global_frame):
    cv2.circle(buffer, (25, 25), 5, red_accent, -1, cv2.LINE_AA)
    cv2.putText(buffer, "CAM-04 : MAIN ATRIUM", (40, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_secondary, 1, cv2.LINE_AA)
    
    cv2.putText(buffer, "O", (w - 50, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "[]", (w - 30, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_secondary, 1, cv2.LINE_AA)
    
    if global_frame is None: return
    ch = h - 50
    if ch <= 0 or w <= 0: return
    
    fh, fw = global_frame.shape[:2]
    scale = min((w-20) / fw, ch / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    if nw <= 0 or nh <= 0: return
    
    resized = cv2.resize(global_frame, (nw, nh))
    ox = (w - nw) // 2
    oy = 40 + (ch - nh) // 2
    buffer[oy:oy+nh, ox:ox+nw] = resized
    
    # Live Overlay Elements
    bx = ox + int(nw * 0.4)
    by = oy + int(nh * 0.2)
    bw = int(nw * 0.25)
    bh = int(nh * 0.55)
    
    cv2.rectangle(buffer, (bx, by), (bx+bw, by+bh), blue_accent, 1)
    cv2.rectangle(buffer, (bx-2, by-2), (bx+3, by+3), blue_accent, -1)
    cv2.rectangle(buffer, (bx+bw-2, by+bh-2), (bx+bw+3, by+bh+3), blue_accent, -1)
    
    draw_rr(buffer, bx, by-22, 80, 18, blue_accent, 4, filled=True)
    cv2.putText(buffer, "ID:492 | 98%", (bx+5, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (25,25,25), 1, cv2.LINE_AA)
    
    cv2.putText(buffer, f"REC: {datetime.now().strftime('%H:%M:%S:%f')[:-4]}", (ox + 20, oy + nh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "FMT: 4K / 60FPS", (ox + 20, oy + nh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_secondary, 1, cv2.LINE_AA)

    
def draw_metrics(buffer, w, h, global_frame):
    cv2.putText(buffer, "System Compute", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
    
    cpu = psutil.cpu_percent() if psutil else 78.0
    mem = psutil.virtual_memory().percent if psutil else 42.0
    gpu = random.randint(85, 95)
    
    def draw_bar(lx, ly, label, val, color):
        cv2.putText(buffer, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_secondary, 1, cv2.LINE_AA)
        cv2.putText(buffer, f"{val}%", (lx + 40, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_primary, 1, cv2.LINE_AA)
        draw_rr(buffer, lx, ly+10, 100, 4, border_color, 2, filled=True)
        draw_rr(buffer, lx, ly+10, int(val), 4, color, 2, filled=True)
        
    draw_bar(20, 60, "CPU", cpu, blue_accent)
    draw_bar(150, 60, "MEM", mem, green_accent)
    draw_bar(280, 60, "GPU", gpu, orange_accent)

def draw_graphs(buffer, w, h, global_frame):
    global last_graph_update, conf_data, prec_data
    if time.time() - last_graph_update > 30:
        conf_data.append(random.uniform(70, 95))
        prec_data.append(random.uniform(60, 85))
        if len(conf_data) > 10: 
            conf_data.pop(0)
            prec_data.pop(0)
        last_graph_update = time.time()
        
    cv2.putText(buffer, "Detection Confidence", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "- Model Conf", (w-180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, blue_accent, 1, cv2.LINE_AA)
    cv2.putText(buffer, "- Precision", (w-90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_secondary, 1, cv2.LINE_AA)
    
    if not conf_data: return
    
    gx, gy = 40, 50
    gw, gh = w - 60, h - 90
    
    # Y-axis Labels
    cv2.putText(buffer, "100", (10, gy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "50", (15, gy+gh//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "0", (20, gy+gh), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    
    for i in range(3):
        y_line = gy + i * (gh//2)
        cv2.line(buffer, (gx, y_line), (gx+gw, y_line), border_color, 1)
        
    # X-axis Labels
    cv2.putText(buffer, "-5m", (gx, gy+gh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "-2.5m", (gx + gw//2 - 15, gy+gh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "Now", (gx+gw-20, gy+gh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    
    def plot_line(data, color, thickness=2):
        if len(data) < 2: return
        pts = []
        for i, val in enumerate(data):
            lx = gx + int(i * (gw / max(1, len(data)-1)))
            ly = gy + gh - int((val / 100.0) * gh)
            pts.append([lx, ly])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(buffer, [pts], False, color, thickness, cv2.LINE_AA)
        
    plot_line(prec_data, text_secondary, 1) 
    plot_line(conf_data, blue_accent, 2) 

def draw_config(buffer, w, h, global_frame):
    cv2.putText(buffer, "Target Tags", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
    
    tags = ['yellow', 'red', 'purple', 'grey', 'orange']
    tag_colors = {
        'yellow': yellow_accent,
        'red': red_accent,
        'purple': purple_accent,
        'grey': grey_accent,
        'orange': orange_accent
    }
    
    sy = 45
    for i, tag in enumerate(tags):
        col = i % 2
        row = i // 2
        px = 15 + col * 85
        py = sy + row * 40
        c = tag_colors[tag]
        
        draw_rr(buffer, px, py, 75, 25, (45, 45, 45), 12, filled=True)
        cv2.circle(buffer, (px + 12, py + 12), 4, c, -1, cv2.LINE_AA)
        
        display_name = tag.capitalize()
        cv2.putText(buffer, display_name, (px + 22, py + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_primary, 1, cv2.LINE_AA)
        
        global target_tag
        if target_tag == tag:
            draw_rr(buffer, px, py, 75, 25, blue_accent, 12, filled=False, thickness=1)
            draw_rr(buffer, px, py, 75, 25, (50, 40, 30), 12, filled=True)
            cv2.circle(buffer, (px + 12, py + 12), 4, c, -1, cv2.LINE_AA)
            cv2.putText(buffer, display_name, (px + 22, py + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_primary, 1, cv2.LINE_AA)

def draw_profile(buffer, w, h, global_frame):
    cv2.putText(buffer, "Last Entity", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
    
    if int(time.time() / 5) % 2 == 0:
        student, tag = "Marcus Thorne", "purple"
    else:
        student, tag = "Sarah Williams", "yellow"
        
    is_ok = (target_tag == tag)
    status = "AUTHORIZED" if is_ok else "DENIED"
    scol = green_accent if is_ok else red_accent
    
    # Avatar
    draw_rr(buffer, 20, 40, 45, 45, (50,50,55), 8, filled=True)
    cv2.circle(buffer, (42, 55), 8, (100, 100, 105), -1, cv2.LINE_AA)
    cv2.ellipse(buffer, (42, 80), (14, 15), 0, 180, 360, (100, 100, 105), -1, cv2.LINE_AA)
    
    cv2.circle(buffer, (65, 80), 5, scol, -1, cv2.LINE_AA)
    
    cv2.putText(buffer, student, (75, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
    cv2.putText(buffer, status, (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, scol, 1, cv2.LINE_AA)
    
    cv2.putText(buffer, "CLEARANCE", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "Level 4" if is_ok else "Level 1", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_primary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "MATCH", (100, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_secondary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "99.8%", (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_primary, 1, cv2.LINE_AA)

def draw_logs(buffer, w, h, global_frame):
    cv2.putText(buffer, "Event Log Stream", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
    cv2.putText(buffer, "Export All", (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, blue_accent, 1, cv2.LINE_AA)
    
    ly = 50
    for i, log in enumerate(event_log):
        if i > 2: break 
        
        parts = log.split(" - ", 1)
        time_str = parts[0]
        msg = parts[1] if len(parts) > 1 else ""
        
        dot_col = text_secondary
        if "DENIED" in msg or "Violation" in msg: dot_col = red_accent
        elif "Target" in msg or "Unrecognized" in msg: dot_col = orange_accent
        elif "Ready" in msg or "AUTHORIZED" in msg: dot_col = green_accent
            
        cv2.circle(buffer, (20, ly+5), 3, dot_col, -1, cv2.LINE_AA)
        cv2.putText(buffer, msg, (35, ly+8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_primary, 1, cv2.LINE_AA)
        cv2.putText(buffer, time_str, (w - 70, ly+8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_secondary, 1, cv2.LINE_AA)
        ly += 35


# ==========================================
# LAYOUT & ENGINE INITIALIZATION
# ==========================================
def init_engine():
    global panels, dock_icons
    panels['camera'] = Panel('camera', 'Feed', draw_camera, 850, 700)
    panels['metrics'] = Panel('metrics', 'Metrics', draw_metrics, 410, 110)
    panels['graphs'] = Panel('graphs', 'Graphs', draw_graphs, 410, 200)
    panels['profile'] = Panel('profile', 'Profile', draw_profile, 195, 170)
    panels['config'] = Panel('config', 'Tags', draw_config, 200, 170)
    panels['logs'] = Panel('logs', 'Logs', draw_logs, 410, 160)
    
    dock_icons.append(DockIcon('camera', 'Feed'))
    dock_icons.append(DockIcon('metrics', 'Metrics'))
    dock_icons.append(DockIcon('graphs', 'Graphs'))
    dock_icons.append(DockIcon('config', 'Tags'))
    dock_icons.append(DockIcon('profile', 'Profile'))
    dock_icons.append(DockIcon('logs', 'Logs'))
    
    dock_w = len(dock_icons) * 70
    start_x = (screen_w - dock_w) // 2 + 35
    for i, icon in enumerate(dock_icons):
        icon.x = start_x + i * 70
        icon.y = screen_h - 75
        panels[icon.panel_id].rect = [icon.x, icon.y, panels[icon.panel_id].tw, panels[icon.panel_id].th]

def calculate_layout():
    rx = screen_w - 450
    ry = 60
    gap = 15
    
    cam = panels['camera']
    if cam.active:
        cam.target_rect[:2] = [250, 60]
        cam.target_rect[2] = rx - 250 - gap
        cam.target_rect[3] = screen_h - 180
    else:
        icon = next(i for i in dock_icons if i.panel_id == 'camera')
        cam.target_rect[:2] = [icon.x, icon.y]
        cam.target_rect[2:] = [0, 0]
        
    for p_id in ['metrics', 'graphs', 'profile', 'config', 'logs']:
        p = panels[p_id]
        if not p.active:
            icon = next(i for i in dock_icons if i.panel_id == p.id)
            p.target_rect[:2] = [icon.x, icon.y]
            p.target_rect[2:] = [0, 0]
            continue
            
        p.target_rect[2:] = [p.tw, p.th]
        if p_id in ['metrics', 'graphs', 'logs']:
            p.target_rect[:2] = [rx, ry]
            ry += p.th + gap
        elif p_id == 'profile':
            p.target_rect[:2] = [rx, ry]
            if not panels['config'].active:
                ry += p.th + gap
        elif p_id == 'config':
            if panels['profile'].active:
                p.target_rect[:2] = [rx + 210, ry]
                ry += p.th + gap
            else:
                p.target_rect[:2] = [rx, ry]
                ry += p.th + gap

def draw_left_sidebar(canvas):
    sx = 30
    cv2.putText(canvas, "UniGuard", (sx, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, blue_accent, 2, cv2.LINE_AA)
    cv2.putText(canvas, "System Node 01", (sx, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_primary, 1, cv2.LINE_AA)
    
    cv2.circle(canvas, (sx+5, 115), 4, green_accent, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Active / Low Latency", (sx+15, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_secondary, 1, cv2.LINE_AA)

    # Blue Overview Button
    draw_rr(canvas, sx, 150, 180, 40, blue_accent, 8, filled=True)
    cv2.putText(canvas, "Overview", (sx+50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
    
    menu_items = ["Live Feed", "Analytics", "Security Log", "Configuration"]
    my = 220
    for item in menu_items:
        cv2.putText(canvas, item, (sx+50, my+25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_primary, 1, cv2.LINE_AA)
        my += 50
        
    # User Profile
    draw_rr(canvas, sx, screen_h - 100, 35, 35, (50, 50, 55), 17, filled=True)
    cv2.putText(canvas, "Admin User", (sx+45, screen_h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_primary, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Access Lvl 4", (sx+45, screen_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_secondary, 1, cv2.LINE_AA)

def cv_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, quit_app, target_tag, event_log
    mouse_x, mouse_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if screen_w - 50 <= x <= screen_w and 0 <= y <= 50:
            quit_app = True
            return

        for icon in dock_icons:
            if icon.x - 20 <= x <= icon.x + icon.w + 20 and icon.y - 20 <= y <= icon.y + icon.w + 20:
                panels[icon.panel_id].active = not panels[icon.panel_id].active
                calculate_layout()
                return
                
        cp = panels['config']
        if cp.active:
            px, py = cp.rect[:2]
            pw, ph = cp.tw, cp.th
            if px <= x <= px+pw and py <= y <= py+ph:
                tags = ['yellow', 'red', 'purple', 'grey', 'orange']
                sy = py + 45
                for i, tag in enumerate(tags):
                    col = i % 2
                    row = i // 2
                    tx = px + 15 + col * 85
                    ty = sy + row * 40
                    if tx <= x <= tx+75 and ty <= y <= ty+25:
                        target_tag = tag
                        event_log.insert(0, f"{datetime.now().strftime('%H:%M:%S')} - Target Tag Set: {tag.capitalize()}")
                        return

def main():
    source = 0 
    if len(sys.argv) > 1 and sys.argv[1].isdigit(): source = int(sys.argv[1])
    cap = cv2.VideoCapture(source)
    
    window_name = 'UniGuard High-Fi OS'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, cv_mouse)
    
    init_engine()
    calculate_layout() 

    while not quit_app:
        ret, frame = cap.read()
        if not ret: break
        
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = bg_color
        
        draw_left_sidebar(canvas)
        
        cv2.rectangle(canvas, (screen_w-40, 0), (screen_w, 40), (40, 30, 30), -1)
        cv2.putText(canvas, "X", (screen_w-25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_accent, 2, cv2.LINE_AA)
        
        for p in panels.values():
            p.update()
            p.render(canvas, frame)
            
        # Dock Background
        dock_w = len(dock_icons) * 70 + 20
        dock_x = (screen_w - dock_w) // 2
        dock_y = screen_h - 95
        draw_rr(canvas, dock_x, dock_y, dock_w, 75, panel_bg, 35, filled=True)
        draw_rr(canvas, dock_x, dock_y, dock_w, 75, border_color, 35, filled=False, thickness=1)
        
        for icon in dock_icons:
            icon.update(mouse_x, mouse_y)
            icon.render(canvas, panels[icon.panel_id].active)
            
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(15) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
