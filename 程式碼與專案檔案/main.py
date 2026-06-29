import os
# 解決 Mac 上 PyTorch 與 MediaPipe 可能造成的 OpenMP 衝突 (Crash)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import math
import numpy as np
import datetime
import threading
from collections import Counter
from ultralytics import YOLO
import mediapipe as mp

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# 計算三個關節點之間的角度
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/math.pi)
    if angle > 180.0: 
        angle = 360 - angle
    return angle

def main():
    print("=========================================")
    print("正在載入 閃電極速版 YOLOv8n 模型 (解決畫面延遲問題)...")
    model = YOLO("yolov8n.pt")
    
    print("正在載入 MediaPipe 姿態模型...")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("正在尋找可用的攝影機 (多執行緒非同步模式)...")
    vs = None
    for cam_index in range(3):
        temp_cap = cv2.VideoCapture(cam_index)
        if temp_cap.isOpened():
            success, _ = temp_cap.read()
            if success:
                temp_cap.release()
                vs = WebcamVideoStream(src=cam_index).start()
                print(f"✅ 成功連線至攝影機頻道 {cam_index} (多執行緒啟動)")
                break
            else:
                temp_cap.release()
        else:
            temp_cap.release()

    if vs is None:
        print("❌ 無法開啟任何攝影機，請檢查權限或設備連接。")
        return

    # 側邊欄寬度設定
    SIDEBAR_W = 320
    
    # 快捷鍵狀態
    show_pose = True
    show_obj = True

    prev_time = time.time()

    print("=========================================")
    print("🚀 超專業級分析系統已啟動！")
    print("   [p] 開關 骨架偵測")
    print("   [o] 開關 物件偵測")
    print("   [s] 截圖儲存 (存至 screenshots 資料夾)")
    print("   [q] 結束程式")
    print("=========================================")

    while True:
        success, frame = vs.read()
        if not success or frame is None: 
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1) # 鏡像翻轉
        H, W, _ = frame.shape
        
        # 建立大畫布 (左邊放攝影機畫面，右邊放側邊欄)
        canvas = np.zeros((H, W + SIDEBAR_W, 3), dtype=np.uint8)

        # 計算系統 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 每個影格重新計算的數據
        inventory = Counter()
        inference_time_ms = 0
        pose_status = "Idle"
        steering_angle = 0

        # --- YOLOv8n 物件偵測 (Object Detection) ---
        if show_obj:
            t1 = time.time()
            # 改用單純的 predict() 替代 track()，徹底解決 Mac 上 tracking 當機卡死與嚴重延遲的問題
            results = model.predict(frame, verbose=False, agnostic_nms=True, conf=0.55)
            inference_time_ms = (time.time() - t1) * 1000

            if len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if len(boxes) > 0:
                    xyxys = boxes.xyxy.int().cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()

                    for xyxy, cls, conf in zip(xyxys, clss, confs):
                        class_name = model.names[cls]
                        inventory[class_name] += 1 # 加進清單統計
                        
                        x1, y1, x2, y2 = xyxy
                        color = (0, 255, 150) if cls == 0 else (255, 150, 50)
                        
                        # 畫出物件追蹤框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 標籤文字底色框
                        label = f"{class_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, max(y1 - 20, 0)), (x1 + tw, max(y1, 20)), color, -1)
                        cv2.putText(frame, label, (x1, max(y1 - 5, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if cls==0 else (255,255,255), 1, cv2.LINE_AA)

        # --- MediaPipe 生物力學姿態分析 (Biomechanics) ---
        if show_pose:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            pose_results = pose.process(img_rgb)
            img_rgb.flags.writeable = True

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                lm = pose_results.pose_landmarks.landmark
                
                # 抓取左右手腕座標 (用來模擬方向盤)
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                l_wrist_px = tuple(np.multiply(l_wrist, [W, H]).astype(int))
                r_wrist_px = tuple(np.multiply(r_wrist, [W, H]).astype(int))
                
                # 畫出虛擬方向盤連線
                cv2.line(frame, l_wrist_px, r_wrist_px, (0, 255, 255), 3)
                
                # 計算方向盤轉動角度 (水平為 0 度)
                dx = r_wrist[0] - l_wrist[0]
                dy = r_wrist[1] - l_wrist[1]
                if dx != 0:
                    steering_angle = math.degrees(math.atan2(dy, dx))
                
                # 畫出中心點與角度
                center_px = ((l_wrist_px[0] + r_wrist_px[0]) // 2, (l_wrist_px[1] + r_wrist_px[1]) // 2)
                cv2.putText(frame, f"{int(steering_angle)} deg", center_px, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

                # 判定動作狀態
                if abs(steering_angle) < 15:
                    pose_status = "Driving Straight"
                elif steering_angle > 15:
                    pose_status = "Turning Right"
                elif steering_angle < -15:
                    pose_status = "Turning Left"

        # === 組合畫面與獨立側邊欄 ===
        canvas[:, :W] = frame
        
        # 繪製側邊欄背景 (深灰黑色)
        sidebar_bg = np.zeros((H, SIDEBAR_W, 3), dtype=np.uint8)
        sidebar_bg[:] = (20, 20, 25) 
        
        # [標題區]
        cv2.putText(sidebar_bg, "CV Analytics Pro", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(sidebar_bg, (20, 55), (SIDEBAR_W-20, 55), (80, 80, 80), 1)

        # [系統效能區]
        cv2.putText(sidebar_bg, "[SYSTEM]", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(sidebar_bg, f"FPS: {fps:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(sidebar_bg, f"Inference: {inference_time_ms:.1f} ms", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(sidebar_bg, f"Model: YOLOv8n (Track)", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # [物件清單統計區]
        cv2.line(sidebar_bg, (20, 185), (SIDEBAR_W-20, 185), (80, 80, 80), 1)
        cv2.putText(sidebar_bg, "[INVENTORY]", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset = 240
        if len(inventory) == 0:
            cv2.putText(sidebar_bg, "No objects detected", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            y_offset += 30
        else:
            for item, count in inventory.most_common():
                cv2.putText(sidebar_bg, f"> {item.capitalize()}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                y_offset += 30

        # [自駕車虛擬方向盤分析]
        cv2.line(sidebar_bg, (20, y_offset + 10), (SIDEBAR_W-20, y_offset + 10), (80, 80, 80), 1)
        cv2.putText(sidebar_bg, "[VIRTUAL STEERING WHEEL]", (20, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(sidebar_bg, f"Steering Angle: {int(steering_angle)} deg", (20, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        status_color = (0, 255, 0) if pose_status == "Driving Straight" else (0, 150, 255)
        cv2.putText(sidebar_bg, f"Status: {pose_status}", (20, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # [快捷鍵提示]
        cv2.putText(sidebar_bg, "Shortcuts: [p] [o] [s] [q]", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # 將側邊欄貼上大畫布
        canvas[:, W:] = sidebar_bg

        # 顯示最終成果
        cv2.imshow("CV Pro Analytics Dashboard", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): show_pose = not show_pose
        elif key == ord('o'): show_obj = not show_obj
        elif key == ord('s'):
            os.makedirs('screenshots', exist_ok=True)
            filename = f"screenshots/pro_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, canvas)
            print(f"📸 專業版截圖已儲存: {filename}")

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
