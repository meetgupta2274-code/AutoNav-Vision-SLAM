from ultralytics import YOLO

class Perception:
    def __init__(self, model_path="yolo11n.pt", min_conf=0.25):
        self.model = YOLO(model_path)
        self.min_conf = min_conf

    def detect_persons(self, frame):
        persons = []
        # We can use model.track for consistent IDs across frames later 
        # For now, standard object detection matching the previous behaviour
        results = self.model(frame, verbose=False)
        for r in results:
            names = r.names
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                
                if label.lower() != "person" or conf < self.min_conf:
                    continue
                
                # Bounding box
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                
                # Check tracking ID if we used track()
                track_id = int(box.id[0]) if box.id is not None else None
                
                persons.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "center": (int(xc), int(yc)),
                    "conf": conf
                })
        return persons

    def track_persons(self, frame):
        """Use object tracking (e.g. ByteTrack) to maintain consistent IDs"""
        persons = []
        results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        for r in results:
            names = r.names
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                
                if label.lower() != "person" or conf < self.min_conf:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                
                track_id = int(box.id[0]) if box.id is not None else None
                
                persons.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "center": (int(xc), int(yc)),
                    "conf": conf
                })
        return persons
