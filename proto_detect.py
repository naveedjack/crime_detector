import sys

try:
    import cv2
    from ultralytics import YOLO
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    print(f"Missing package: {e.name}. Please install it using pip.")
    sys.exit(1)

# Load YOLOv8 model

model = YOLO("yolov8n.pt")
# Only keep these classes for detection
CRIME_CLASSES = {"person", "knife", "scissors", "gun", "pistol", "rifle"}
WEAPON_CLASSES = {"knife", "scissors", "gun", "pistol", "rifle"}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def send_warning(label, frame, reason="Weapon Detected"):
    print(f"WARNING: {reason} - {label}")
    cv2.imwrite("crime_scene.jpg", frame)

class LandmarkHistory:
    def __init__(self):
        self.prev = []

    def update(self, current):
        self.prev = current.copy() if current else []

    def get(self):
        return self.prev

def get_xy(lm, idx):
    return np.array([lm[idx].x, lm[idx].y])

def detect_specific_offensive_action(landmarks_list, prev_landmarks_list):
    # Further reduced sensitivity: require even closer and faster for fighting
    violence_score = 0
    violence_type = None
    for i, lm1 in enumerate(landmarks_list):
        if lm1 is None:
            continue
        prev1 = prev_landmarks_list[i] if prev_landmarks_list and len(prev_landmarks_list) > i else None
        for j, lm2 in enumerate(landmarks_list):
            if i == j or lm2 is None:
                continue
            # Punching: hand moves rapidly toward another's head/neck
            for wrist_idx in [mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.LEFT_WRIST.value]:
                hand = get_xy(lm1, wrist_idx)
                prev_hand = get_xy(prev1, wrist_idx) if prev1 else hand
                nose2 = get_xy(lm2, mp_pose.PoseLandmark.NOSE.value)
                lsh2 = get_xy(lm2, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                rsh2 = get_xy(lm2, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                neck2 = (lsh2 + rsh2) / 2
                hand_vel = np.linalg.norm(hand - prev_hand)
                dist_head = np.linalg.norm(hand - nose2)
                dist_neck = np.linalg.norm(hand - neck2)
                # Require very close and fast for fighting
                if ((dist_head < 0.045 or dist_neck < 0.045) and hand_vel > 0.06):
                    violence_score = max(violence_score, 95)
                    violence_type = "fighting"
                elif ((dist_head < 0.07 or dist_neck < 0.07) and hand_vel > 0.03):
                    violence_score = max(violence_score, 60)
                    violence_type = "fighting"
            # Grabbing: hand stays close to neck/collar
            for wrist_idx in [mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.LEFT_WRIST.value]:
                hand = get_xy(lm1, wrist_idx)
                lsh2 = get_xy(lm2, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                rsh2 = get_xy(lm2, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                neck2 = (lsh2 + rsh2) / 2
                dist_neck = np.linalg.norm(hand - neck2)
                if dist_neck < 0.025:
                    violence_score = max(violence_score, 90)
                    violence_type = "fighting"
                elif dist_neck < 0.045:
                    violence_score = max(violence_score, 55)
                    violence_type = "fighting"
            # Kicking: foot moves rapidly toward another's torso/legs
            for foot_idx in [mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value]:
                foot = get_xy(lm1, foot_idx)
                prev_foot = get_xy(prev1, foot_idx) if prev1 else foot
                lhip2 = get_xy(lm2, mp_pose.PoseLandmark.LEFT_HIP.value)
                rhip2 = get_xy(lm2, mp_pose.PoseLandmark.RIGHT_HIP.value)
                midhip2 = (lhip2 + rhip2) / 2
                foot_vel = np.linalg.norm(foot - prev_foot)
                dist_hip = np.linalg.norm(foot - midhip2)
                if dist_hip < 0.055 and foot_vel > 0.06:
                    violence_score = max(violence_score, 95)
                    violence_type = "fighting"
                elif dist_hip < 0.09 and foot_vel > 0.03:
                    violence_score = max(violence_score, 60)
                    violence_type = "fighting"
            # Dashing: body moves rapidly toward another person
            mid1 = (get_xy(lm1, mp_pose.PoseLandmark.LEFT_HIP.value) + get_xy(lm1, mp_pose.PoseLandmark.RIGHT_HIP.value)) / 2
            prev_mid1 = (get_xy(prev1, mp_pose.PoseLandmark.LEFT_HIP.value) + get_xy(prev1, mp_pose.PoseLandmark.RIGHT_HIP.value)) / 2 if prev1 else mid1
            mid2 = (get_xy(lm2, mp_pose.PoseLandmark.LEFT_HIP.value) + get_xy(lm2, mp_pose.PoseLandmark.RIGHT_HIP.value)) / 2
            body_vel = np.linalg.norm(mid1 - prev_mid1)
            dist_body = np.linalg.norm(mid1 - mid2)
            if dist_body < 0.07 and body_vel > 0.045:
                violence_score = max(violence_score, 85)
                violence_type = "fighting"
            elif dist_body < 0.12 and body_vel > 0.02:
                violence_score = max(violence_score, 50)
                violence_type = "fighting"
    if violence_score >= 80:
        return True, violence_type, violence_score
    elif violence_score >= 50:
        return None, "suspicious activity", violence_score
    else:
        return False, None, 0

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found or cannot be opened!")
        return

    landmark_history = LandmarkHistory()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        results = model(frame, imgsz=640, conf=0.35)
        boxes = []
        labels = []
        weapon_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                # Only process relevant crime-related classes
                if label.lower() not in CRIME_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                labels.append(label)

        # Check for weapons
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            if label.lower() in WEAPON_CLASSES:
                weapon_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "WARNING: WEAPON!", (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                send_warning("weapon", frame, reason="Weapon Detected")
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If no weapon, check for specific offensive actions using pose
        if not weapon_detected:
            landmarks_list = []
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                if label == "person":
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size == 0:
                        landmarks_list.append(None)
                        continue
                    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                    result = pose.process(person_rgb)
                    if result.pose_landmarks:
                        landmarks_list.append(result.pose_landmarks.landmark)
                    else:
                        landmarks_list.append(None)
            prev_landmarks_list = landmark_history.get() if hasattr(landmark_history, 'get') else None
            if len(landmarks_list) > 1:
                detected, action_type, violence_score = detect_specific_offensive_action(landmarks_list, prev_landmarks_list)
                if detected:
                    # High violence: red, label as FIGHTING
                    for (x1, y1, x2, y2), label in zip(boxes, labels):
                        if label == "person":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"VIOLENCE: FIGHTING ({violence_score})", (x1, y1-20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    send_warning("fighting", frame, reason=f"Fighting Detected (Score: {violence_score})")
                elif action_type is not None and violence_score > 0:
                    # Suspicious/uncertain: yellow, label as SUSPICIOUS ACTIVITY
                    for (x1, y1, x2, y2), label in zip(boxes, labels):
                        if label == "person":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            cv2.putText(frame, f"SUSPICIOUS ACTIVITY ({violence_score})", (x1, y1-20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                # else: no violence
            landmark_history.update(landmarks_list)

        cv2.imshow("Crime Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()