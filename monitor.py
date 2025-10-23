import cv2
import time

# ==== 配置区域 ====
rtsp_url = "rtsp://admin:kemande1206@192.168.121.153:554/live" # 修改为你的 RTSP 地址
output_path = "output.avi"                                  # 输出文件路径
duration = 60                                               # 录制时长（秒）

# 打开 RTSP 流
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("❌ 无法打开 RTSP 流，请检查地址或网络。")
    exit()

# 获取视频属性（可能有的摄像头不提供，设置默认值）
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None or fps != fps:  # NaN 检查
    fps = 25  # 默认帧率
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

# 定义视频编码器和输出文件对象
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # avi格式常用编码
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"✅ 开始录制（时长 {duration} 秒，输出文件：{output_path}）...")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取帧，停止录制。")
        break

    # 写入视频帧
    out.write(frame)

    # 显示画面
    # cv2.imshow("RTSP Stream Recording", frame)

    # 检查时长
    if time.time() - start_time >= duration:
        print("⏹️ 录制完成。")
        break

    # 按下 q 可提前退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹️ 用户终止录制。")
        break

cap.release()
out.release()
cv2.destroyAllWindows()