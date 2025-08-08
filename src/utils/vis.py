import cv2

def draw_box_id(frame, box, tid, color=(0,255,0), role_text=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    label = f"ID {tid}" if tid is not None else ""
    if role_text:
        label += f" | {role_text}"
    if label:
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def put_text(frame, text, org=(10,30)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
