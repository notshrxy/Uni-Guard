"""
End-to-End Inference Pipeline
Main orchestrator for end-to-end pipeline.
This script runs your camera, detects persons/cards/shoes using YOLO
Combines CA-YOLOv8 detection + compliance check + face recognition (If properly planned with facial database)
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import ctypes
from ctypes import wintypes

# Global UI Variables
ui_dragging = False
ui_offset_x, ui_offset_y = 0, 0
ui_hwnd = None
ui_quit_app = False
ui_take_snapshot = False
ui_border_removed = False

def ui_mouse_callback(event, x, y, flags, param):
    global ui_dragging, ui_offset_x, ui_offset_y, ui_hwnd, ui_quit_app, ui_take_snapshot
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 30 and 10 <= y <= 30:
            ui_quit_app = True
        elif 30 <= x <= 50 and 10 <= y <= 30:
            ctypes.windll.user32.ShowWindow(ui_hwnd, 6) # SW_MINIMIZE
        elif 50 <= x <= 70 and 10 <= y <= 30:
            ui_take_snapshot = True
        elif y <= 40:
            ui_dragging = True
            pt = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            rect = wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(ui_hwnd, ctypes.byref(rect))
            ui_offset_x = pt.x - rect.left
            ui_offset_y = pt.y - rect.top

    elif event == cv2.EVENT_MOUSEMOVE:
        if ui_dragging and ui_hwnd:
            pt = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            new_x = pt.x - ui_offset_x
            new_y = pt.y - ui_offset_y
            ctypes.windll.user32.SetWindowPos(ui_hwnd, 0, new_x, new_y, 0, 0, 0x0001 | 0x0004 | 0x0010)

    elif event == cv2.EVENT_LBUTTONUP:
        ui_dragging = False

def ui_remove_windows_border(window_name, w, h):
    global ui_hwnd
    ui_hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if not ui_hwnd: return
    
    GWL_STYLE = -16
    WS_CAPTION = 0x00C00000
    WS_THICKFRAME = 0x00040000
    WS_MINIMIZEBOX = 0x00020000
    WS_MAXIMIZEBOX = 0x00010000
    WS_SYSMENU = 0x00080000
    
    style = ctypes.windll.user32.GetWindowLongW(ui_hwnd, GWL_STYLE)
    style = style & ~WS_CAPTION & ~WS_THICKFRAME & ~WS_MINIMIZEBOX & ~WS_MAXIMIZEBOX & ~WS_SYSMENU
    ctypes.windll.user32.SetWindowLongW(ui_hwnd, GWL_STYLE, style)
    
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)
    pos_x = max(0, (screen_w - w) // 2)
    pos_y = max(0, (screen_h - h) // 2)
    
    ctypes.windll.user32.SetWindowPos(ui_hwnd, 0, pos_x, pos_y, 0, 0, 0x0001 | 0x0004 | 0x0020)
    
    hrgn = ctypes.windll.gdi32.CreateRoundRectRgn(0, 0, w, h, 20, 20)
    ctypes.windll.user32.SetWindowRgn(ui_hwnd, hrgn, True)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class SmartIDPipeline:
    """Complete pipeline: Detection → Compliance → Face Recognition → Alerts."""

    def __init__(self, model_path, face_db_dir='face_db', conf_threshold=0.25,
                 card_overlap_threshold=0.1, face_similarity_threshold=0.5):
        """
        Args:
            model_path: Path to trained CA-YOLOv8 .pt model file.
            face_db_dir: Directory containing face database.
            conf_threshold: Detection confidence threshold.
            card_overlap_threshold: IoU threshold for card-person association.
            face_similarity_threshold: Cosine similarity threshold for face matching.
        """
        self.conf_threshold = conf_threshold
        self.card_overlap_threshold = card_overlap_threshold
        self.alerts = []

        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"[Pipeline] Loaded custom model: {model_path}")
        
        # Load official pre-trained COCO model for flawless, stable person detection
        self.coco_model = YOLO("yolov8n.pt")
        print("[Pipeline] Loaded official pre-trained COCO model yolov8n.pt for ultra-stable person detection!")

        # Load face recognition pipeline
        from facial_features.facial_pipeline import FacePipeline
        from facial_features.facial_database import FaceDatabase

        self.face_pipeline = FacePipeline(similarity_threshold=face_similarity_threshold)
        self.face_db = FaceDatabase(db_dir=face_db_dir, similarity_threshold=face_similarity_threshold)

    def detect(self, frame):
        """Run dual-model detection: COCO model for stable persons, custom model for cards/shoes."""
        # 1. Run official pre-trained model for flawless person detection (conf >= 0.5)
        coco_results = self.coco_model(frame, conf=0.5, verbose=False)[0]
        persons = []
        for box in coco_results.boxes:
            cls = int(box.cls[0])
            cls_name = self.coco_model.names.get(cls, "").lower()
            if cls_name == "person":
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                persons.append([*xyxy, conf])

        # 2. Run custom model for ID card/lanyard and shoes
        custom_results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        cards = []
        shoes = []
        for box in custom_results.boxes:
            cls = int(box.cls[0])
            cls_name = self.model.names.get(cls, "").lower()
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            entry = [*xyxy, conf]
            
            # Ignore person detections from custom model (COCO is infinitely better)
            if "card" in cls_name:
                cards.append(entry)
            elif "shoes" in cls_name or "shoe" in cls_name:
                shoes.append(entry)

        return {'persons': persons, 'cards': cards, 'shoes': shoes}

    def check_compliance(self, persons, cards, shoes):
        """Check which persons are wearing ID cards and shoes."""
        results = []
        used_cards = set()
        used_shoes = set()

        for person in persons:
            px1, py1, px2, py2, p_conf = person
            person_h = py2 - py1

            card_zone_y2 = py1 + 0.85 * person_h
            shoe_zone_y1 = py2 - 0.45 * person_h
            shoe_zone_y2 = py2 + 0.15 * person_h

            has_id = False
            has_shoes = False
            matched_card = None
            matched_shoe = None

            for i, card in enumerate(cards):
                if i in used_cards: continue
                cx1, cy1, cx2, cy2, c_conf = card
                card_cx, card_cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                if (px1 - 25 <= card_cx <= px2 + 25 and py1 <= card_cy <= card_zone_y2):
                    has_id = True
                    matched_card = card
                    used_cards.add(i)
                    break

            for i, shoe in enumerate(shoes):
                if i in used_shoes: continue
                sx1, sy1, sx2, sy2, s_conf = shoe
                shoe_cx, shoe_cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
                if (px1 - 25 <= shoe_cx <= px2 + 25 and shoe_zone_y1 <= shoe_cy <= shoe_zone_y2):
                    has_shoes = True
                    matched_shoe = shoe
                    used_shoes.add(i)
                    break

            results.append({
                'box': person[:4],
                'confidence': person[4],
                'has_id': has_id,
                'has_shoes': has_shoes,
                'card_box': matched_card[:4] if matched_card else None,
                'shoe_box': matched_shoe[:4] if matched_shoe else None,
            })

        return results

    def _log_attendance(self, identity, tag_verified=None, violation=None):
        """Log attendance to SQLite with a 10-second debounce per student,
        short-circuiting the debounce if a student transitions from VIOLATION to PRESENT (Self-Correction)."""
        roll_no = identity.get('person_id')
        if not roll_no:
            return False
            
        import time
        current_time = time.time()
        
        if not hasattr(self, '_last_log_times'):
            self._last_log_times = {}
        if not hasattr(self, '_last_log_status'):
            self._last_log_status = {}
            
        status = "PRESENT" if not violation else "VIOLATION"
        
        # Check if this is an immediate transition from VIOLATION to PRESENT (Self-Correction)
        is_upgrade = (self._last_log_status.get(roll_no) == "VIOLATION" and status == "PRESENT")
        
        if not is_upgrade:
            last_log = self._last_log_times.get(roll_no, 0)
            if current_time - last_log < 10:
                return False
                
        self._last_log_times[roll_no] = current_time
        self._last_log_status[roll_no] = status
        
        self.face_db.sql_db.log_attendance(
            roll_no=roll_no,
            status=status,
            tag_color=tag_verified,
            violation=violation
        )
        print(f"[Pipeline] SQL Logged: {identity['name']} - {status}")
        return True

    def _get_tag_color(self, frame, card_box):
        """Detect dominant color of ID tag by looking at the detected lanyard bounding box."""
        x1, y1, x2, y2 = map(int, card_box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return "unknown"
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ranges = {
            'red': [([0, 90, 50], [10, 255, 255]), ([170, 90, 50], [180, 255, 255])],
            'yellow': [([15, 90, 80], [40, 255, 255])],
            'purple': [([110, 70, 30], [170, 255, 255])]
        }
        
        results = {}
        for color, color_ranges in ranges.items():
            mask = None
            for (lower, upper) in color_ranges:
                m = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            results[color] = cv2.countNonZero(mask)
            
        self.last_color_debug = results
        print(f"[Color Debug] Pixel Counts -> Red: {results['red']} | Yellow: {results['yellow']} | Purple: {results['purple']}")
        best_color = max(results, key=results.get)
        if results[best_color] < 50:
            return "unknown"
        return best_color

    def apply_clahe(self, img):
        """Enhance local contrast and brighten shadows using CLAHE on LAB space."""
        import cv2
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame, target_tag='purple'):
        """Process a single frame with target batch awareness."""
        self.last_color_debug = None
        frame = self.apply_clahe(frame)
        detections = self.detect(frame)
        person_checks = self.check_compliance(detections['persons'], detections['cards'], detections['shoes'])

        self.frame_count = getattr(self, 'frame_count', 0) + 1
        
        compliance_results = []
        violations = []
        permitted_scans = []
        
        import time
        now = time.time()
        for i, check in enumerate(person_checks):
            identity = {'name': 'Searching...', 'gender': 'unknown'}
            
            cache_key = f"person_{i}"
            if not hasattr(self, 'cached_identities'):
                self.cached_identities = {}
                
            # If we haven't recognized the person yet, keep looking every frame.
            # Once recognized, throttle/cache the facial model to run only once every 8 frames to eliminate lag.
            run_face = (self.frame_count % 8 == 0) or (cache_key not in self.cached_identities) or (self.cached_identities[cache_key].get('name', 'Searching...') == 'Searching...')
            
            if run_face:
                face_result = self.face_pipeline.process_violator(frame, check['box'])
                if face_result:
                    match = self.face_db.identify(face_result['embedding'])
                    if match:
                        identity = match
                self.cached_identities[cache_key] = identity
            else:
                identity = self.cached_identities.get(cache_key, identity)
            
            roll_no = identity.get('person_id')
            is_already_present_today = False
            
            if roll_no:
                if not hasattr(self, '_student_sessions'):
                    self._student_sessions = {}
                    
                if not hasattr(self, '_present_students_today'):
                    self._present_students_today = set()
                    
                # 1. Start or retrieve the scanning session
                if roll_no not in self._student_sessions or (now - self._student_sessions[roll_no]['last_seen'] > 4.0):
                    # Query SQL database ONCE at the start of their walk-up session!
                    db_present = False
                    if roll_no in self._present_students_today:
                        db_present = True
                    else:
                        if hasattr(self.face_db, 'sql_db'):
                            conn = None
                            try:
                                conn = self.face_db.sql_db._get_connection()
                                cursor = conn.cursor()
                                today_date = datetime.now().strftime('%Y-%m-%d')
                                cursor.execute('''
                                    SELECT status FROM attendance
                                    WHERE student_id = ? AND date(timestamp) = ? AND status = 'PRESENT'
                                ''', (roll_no, today_date))
                                if cursor.fetchone() is not None:
                                    self._present_students_today.add(roll_no)
                                    db_present = True
                            except Exception as e:
                                print(f"[Pipeliner] Error checking daily attendance status: {e}")
                            finally:
                                if conn:
                                    conn.close()
                                    
                    self._student_sessions[roll_no] = {
                        'first_seen': now,
                        'last_seen': now,
                        'history': [],
                        'logged_status': 'PRESENT' if db_present else None,
                        'completed': db_present,
                        'already_present_today': db_present
                    }
                else:
                    self._student_sessions[roll_no]['last_seen'] = now
                    
                session = self._student_sessions[roll_no]
                is_already_present_today = session.get('already_present_today', False)
                
            if is_already_present_today:
                detected_tag = target_tag
                is_compliant = True
                v_types = []
                
                # Make sure the session remains in PRESENT state
                session['logged_status'] = 'PRESENT'
                session['completed'] = True
            else:
                detected_tag = "none"
                if check['card_box'] is not None:
                    detected_tag = self._get_tag_color(frame, check['card_box'])
                
                has_id = check['has_id']
                # unused after shoes requirement commented out
                # has_shoes = check['has_shoes']
                # is_female = identity.get('gender') == 'female'
                
                tag_matches = (detected_tag == target_tag)
                # Mandatory shoes logic commented out - both genders only require valid ID cards
                # is_compliant = has_id and tag_matches and (is_female or has_shoes)
                is_compliant = has_id and tag_matches
                
                v_types = []
                if not has_id: v_types.append("No ID")
                elif not tag_matches: v_types.append(f"Wrong Batch (Got {detected_tag})")
                # if not is_female and not has_shoes: v_types.append("No Shoes")
                
                # Temporal smoothing and stabilization (2-second sliding window validation)
                if roll_no:
                    # Append current frame raw prediction
                    session['history'].append((now, is_compliant, detected_tag, v_types))
                    
                    # Prune history to keep only the last 2.0 seconds
                    session['history'] = [h for h in session['history'] if now - h[0] <= 2.0]
                    
                    # Check if we are still stabilizing (first 1.0 second of scanning)
                    if now - session['first_seen'] < 1.0:
                        is_compliant = True
                        v_types = []
                        detected_tag = target_tag
                    else:
                        history_len = len(session['history'])
                        compliant_count = sum(1 for h in session['history'] if h[1])
                        compliance_ratio = compliant_count / history_len if history_len > 0 else 0.0
                        
                        if compliance_ratio >= 0.40:
                            is_compliant = True
                            v_types = []
                            # Retrieve dominant tag in sliding history
                            from collections import Counter
                            tags = [h[2] for h in session['history'] if h[2] != 'none']
                            if tags:
                                detected_tag = Counter(tags).most_common(1)[0][0]
                            else:
                                detected_tag = target_tag
                        else:
                            is_compliant = False
                            # Retrieve dominant violation types and tag in sliding history
                            from collections import Counter
                            all_v_types = []
                            for h in session['history']:
                                all_v_types.extend(h[3])
                            if all_v_types:
                                v_types = list(set(all_v_types))
                            else:
                                v_types = ["No ID"]
                                
                            tags = [h[2] for h in session['history'] if h[2] != 'none']
                            if tags:
                                detected_tag = Counter(tags).most_common(1)[0][0]
                                
            # If session is locked/completed with PRESENT, force compliant state display
            is_completed_present = (roll_no and self._student_sessions.get(roll_no, {}).get('logged_status') == "PRESENT")
            if is_completed_present:
                is_compliant = True
                v_types = []
                
            result = {
                **check,
                'identity': identity,
                'detected_tag': detected_tag,
                'compliant': is_compliant,
                'violation_type': v_types,
                'session_logged_status': self._student_sessions.get(roll_no, {}).get('logged_status') if roll_no else None
            }
            
            compliance_results.append(result)
            
            logged_to_db = False
            if identity.get('person_id') and not is_already_present_today:
                # Only log to database after the 2-second stabilization window is complete!
                if now - session['first_seen'] >= 2.0:
                    current_status = "PRESENT" if not v_types else "VIOLATION"
                    
                    # Log-Once rules per scan session:
                    already_present = (session.get('logged_status') == "PRESENT")
                    already_violation = (session.get('logged_status') == "VIOLATION")
                    
                    should_log = False
                    if not already_present:
                        # Log if we haven't logged anything yet, OR if we are upgrading a violation to present
                        if not already_violation or (already_violation and current_status == "PRESENT"):
                            should_log = True
                            
                    if should_log:
                        v_str = ", ".join(v_types) if v_types else None
                        logged_to_db = self._log_attendance(identity, tag_verified=detected_tag, violation=v_str)
                        if logged_to_db:
                            session['logged_status'] = current_status

            if logged_to_db and is_compliant:
                permitted_scans.append(result)

            if not is_compliant:
                # Debounce UI Event Log alerts strictly to prevent rapid frame-level spam!
                should_trigger_ui_alert = False
                
                # STRICT RULE: Never trigger compliance violation alerts for transient "Searching..." state!
                if identity.get('name') not in ["Searching...", "Searching", ""]:
                    if identity.get('person_id'):
                        # For known students, only alert if we successfully wrote a new debounced SQL log
                        if logged_to_db:
                            should_trigger_ui_alert = True
                    else:
                        # For unknown visitors, debounce using a spatially-rounded grid key to absorb coordinate jitter!
                        if not hasattr(self, '_last_ui_violation_times'):
                            self._last_ui_violation_times = {}
                        px1, py1, px2, py2 = check['box']
                        # Round to nearest 100 pixels to group jittering bounding boxes
                        grid_x = int(px1 / 100) * 100
                        grid_y = int(py1 / 100) * 100
                        visitor_key = f"visitor_{grid_x}_{grid_y}"
                        
                        last_ui_log = self._last_ui_violation_times.get(visitor_key, 0)
                        if now - last_ui_log >= 10:
                            self._last_ui_violation_times[visitor_key] = now
                            should_trigger_ui_alert = True
                
                if should_trigger_ui_alert:
                    violations.append(result)

        annotated = self._draw_results(frame.copy(), detections, compliance_results)
        return {
            'detections': detections,
            'compliance': compliance_results,
            'violations': violations,
            'permitted': permitted_scans,
            'color_debug': self.last_color_debug,
            'annotated_frame': annotated,
        }

    def _draw_results(self, frame, detections, compliance_results):
        """Draw bounding boxes and labels."""
        for card in detections['cards']:
            x1, y1, x2, y2, _ = card
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
        for shoe in detections['shoes']:
            x1, y1, x2, y2, _ = shoe
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        for result in compliance_results:
            x1, y1, x2, y2 = map(int, result['box'])
            id_info = result['identity']
            name = id_info.get('name', 'Unknown')

            session_status = result.get('session_logged_status')

            if session_status == "PRESENT":
                color = (0, 255, 0)
                label = f"{name} - VERIFIED PRESENT"
            elif session_status == "VIOLATION":
                color = (0, 0, 255)
                label = f"{name} - LOCKED (ABSENT)"
            else:
                if result['compliant']:
                    color = (0, 255, 0)
                    label = f"{name} - OK"
                else:
                    color = (0, 0, 255)
                    v_str = "/".join(result['violation_type']) if result['violation_type'] else "Violation"
                    label = f"{name} - {v_str}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def _draw_macos_window(self, frame):
        """Draws a fake Mac OS title bar on top of the frame for presentation purposes."""
        h, w = frame.shape[:2]
        bar_height = 40
        
        # Create a new image with extra space for the title bar
        window = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
        
        # Draw the title bar (light gray)
        window[0:bar_height, 0:w] = (235, 235, 235)
        
        # Draw the buttons (Red, Yellow, Green)
        radius = 6
        spacing = 20
        y_center = bar_height // 2
        cv2.circle(window, (spacing, y_center), radius, (76, 76, 255), -1, lineType=cv2.LINE_AA)      # Red (BGR)
        cv2.circle(window, (spacing * 2, y_center), radius, (51, 204, 255), -1, lineType=cv2.LINE_AA)   # Yellow
        cv2.circle(window, (spacing * 3, y_center), radius, (76, 217, 40), -1, lineType=cv2.LINE_AA)    # Green
        
        # Draw title text
        title = "UniGuard - Live Compliance Monitor"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = y_center + text_size[1] // 2
        cv2.putText(window, title, (text_x, text_y), font, font_scale, (100, 100, 100), thickness, lineType=cv2.LINE_AA)
        
        # Place the original frame below the title bar
        window[bar_height:h + bar_height, 0:w] = frame
        
        return window

    def process_video(self, source=0, display=True):
        global ui_quit_app, ui_take_snapshot, ui_border_removed
        cap = cv2.VideoCapture(source)
        
        if display:
            cv2.namedWindow('UniGuard Live', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('UniGuard Live', ui_mouse_callback)
            ui_quit_app = False
            ui_take_snapshot = False
            ui_border_removed = False

        while cap.isOpened() and not ui_quit_app:
            ret, frame = cap.read()
            if not ret: break
            result = self.process_frame(frame)
            
            if display:
                macos_frame = self._draw_macos_window(result['annotated_frame'])
                
                if ui_take_snapshot:
                    os.makedirs("snapshots", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshots/violation_{timestamp}.jpg"
                    cv2.imwrite(filename, macos_frame)
                    print(f"\n[Snapshot] Saved violation evidence to {filename}")
                    ui_take_snapshot = False
                    
                cv2.imshow('UniGuard Live', macos_frame)
                
                if not ui_border_removed:
                    frame_h, frame_w = macos_frame.shape[:2]
                    ui_remove_windows_border('UniGuard Live', frame_w, frame_h)
                    ui_border_removed = True
                    
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source', type=str, default='0')
    args = parser.parse_args()
    
    pipeline = SmartIDPipeline(model_path=args.model)
    pipeline.process_video(source=int(args.source) if args.source.isdigit() else args.source)