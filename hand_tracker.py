import cv2
import mediapipe as mp
import time
import os

# 2026 最新版 API 调用方式
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 1. 自动下载模型文件 (如果下载慢，我会教你手动下)
import urllib.request
model_path = 'hand_landmarker.task'
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# 增加判断：如果文件不存在才下载
if not os.path.exists(model_path):
    print("正在首次下载模型，请稍候...")
    try:
        urllib.request.urlretrieve(url, model_path)
        print("模型下载完成！")
    except Exception as e:
        print(f"下载失败: {e}")
else:
    print("检测到本地已存在模型，直接加载。")

# 2. 手势判断逻辑
def get_gesture_name(landmarks):
    # 计算食指、中指、无名指、小指是否伸直
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18] # 第二关节
    fingers = []
    for t, p in zip(tips, pip_joints):
        if landmarks[t].y < landmarks[p].y: # Y轴越小位置越高
            fingers.append(1)
        else:
            fingers.append(0)
    
    res = sum(fingers)
    if res == 0: return "Fist (Quan)"
    if res == 4: return "Open (Bu)"
    if res == 2 and fingers[0] == 1 and fingers[1] == 1: return "Victory (Scissors)"
    return "Detecting..."

# 3. 初始化检测器
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

with HandLandmarker.create_from_options(options) as landmarker:
    #选择摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 转换为 MediaPipe 图像格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 获取当前时间戳（毫秒）
        timestamp = int(time.time() * 1000)
        
        # 执行检测
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                # 绘制关键点（手动绘制，避免 solutions 报错）
                for lm in landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # 获取并显示手势名称
                gesture = get_gesture_name(landmarks)
                cv2.putText(frame, gesture, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('MediaPipe 2026 Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()