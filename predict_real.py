import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import decord
from pytorchvideo.models.hub import slowfast_r50
import torchvision.transforms as T
import os
import logging
import time
import threading
from collections import deque
import queue
import datetime

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlowFastPredictor:
    """SlowFast视频分类预测器 - 支持实时RTSP流和双视频保存"""

    def __init__(self, model_path: str, num_classes: int = 4, device: str = "auto",
                 confidence_threshold: float = 0.6, save_video: bool = True,
                 video_save_dir: str = "saved_videos", video_duration: int = 300):
        """
        初始化SlowFast预测器

        Args:
            model_path: 训练好的模型权重路径
            num_classes: 类别数量
            device: 运行设备 ('auto', 'cuda', 'cpu')
            confidence_threshold: 置信度阈值，低于此值分类为None
            save_video: 是否保存视频
            video_save_dir: 视频保存目录
            video_duration: 单个视频时长（秒）
        """
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.num_frames = 32
        self.confidence_threshold = confidence_threshold
        self.save_video = save_video
        self.video_save_dir = video_save_dir
        self.video_duration = video_duration

        # 创建视频保存目录
        if self.save_video:
            os.makedirs(os.path.join(self.video_save_dir, "original"), exist_ok=True)
            os.makedirs(os.path.join(self.video_save_dir, "annotated"), exist_ok=True)

        # 使用线程安全的队列 - 增加队列大小减少丢帧
        self.frame_buffer = deque(maxlen=self.num_frames * 2)
        self.prediction_queue = queue.Queue(maxsize=3)  # 减少预测队列大小，避免堆积
        self.original_video_queue = queue.Queue(maxsize=30)  # 增加视频队列大小
        self.annotated_video_queue = queue.Queue(maxsize=30)

        self.latest_result = None
        self.is_running = False
        self.capture_thread = None
        self.prediction_thread = None
        self.original_video_thread = None
        self.annotated_video_thread = None

        # 视频保存相关变量
        self.current_original_writer = None
        self.current_annotated_writer = None
        self.current_video_filename = None
        self.video_start_time = None
        self.video_segment_count = 0
        self.frame_width = 640  # 默认值
        self.frame_height = 480  # 默认值
        self.fps = 15

        # 性能统计和同步
        self.frame_count = 0
        self.prediction_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_prediction_time = 0
        self.prediction_interval = 0.5  # 预测间隔，减少计算负载

        # 检查模型文件
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 预处理变换 - 与训练时保持一致
        self.transform = self._get_transform()

        logger.info(f"SlowFast预测器初始化完成，使用设备: {self.device}, 置信度阈值: {confidence_threshold}, 视频保存: {save_video}")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        try:
            # 创建模型结构 - 与训练时相同
            model = SlowFastModel(self.num_classes, pretrained=False)

            if model_path:  # 只有提供了模型路径时才加载权重
                # 加载权重
                checkpoint = torch.load(model_path, map_location=self.device)

                # 处理多GPU训练保存的权重
                state_dict = self._process_checkpoint(checkpoint)

                # 加载权重
                model.load_state_dict(state_dict)
                model.to(self.device)

                logger.info("模型加载成功")
            else:
                logger.info("使用预训练模型")
                model = SlowFastModel(self.num_classes, pretrained=True)
                model.to(self.device)

            return model

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 返回一个预训练模型作为备选
            logger.info("使用预训练模型作为备选")
            model = SlowFastModel(self.num_classes, pretrained=True)
            model.to(self.device)
            return model

    def _process_checkpoint(self, checkpoint):
        """处理检查点文件"""
        if isinstance(checkpoint, dict):
            # 如果是字典，可能是状态字典或其他格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            # 如果是模型实例
            state_dict = checkpoint.state_dict()
        else:
            # 假设已经是状态字典
            state_dict = checkpoint

        # 移除module前缀（如果使用DataParallel训练）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]
            else:
                new_key = k
            new_state_dict[new_key] = v

        return new_state_dict

    def _get_transform(self):
        """获取预处理变换 - 与训练时保持一致"""
        return T.Compose([
            T.Resize(256),
            lambda x: x / 255.0,
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_frames(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """预处理视频帧"""
        if len(frames) == 0:
            raise ValueError("输入帧列表为空")

        # 如果帧数不足，重复最后一帧
        if len(frames) < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))

        # 采样策略：从视频中均匀采样num_frames帧
        total_frames = len(frames)
        if total_frames > self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        # 转换为tensor并预处理
        frame_tensors = []
        for frame in sampled_frames:
            try:
                # 确保帧是RGB格式
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # 转换为tensor并调整维度 [H, W, C] -> [C, H, W]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                    frame_tensor = self.transform(frame_tensor)
                    frame_tensors.append(frame_tensor)
            except Exception as e:
                logger.warning(f"帧处理失败: {e}")
                continue

        if len(frame_tensors) < self.num_frames:
            raise ValueError(f"有效帧数量不足: {len(frame_tensors)}/{self.num_frames}")

        # 堆叠帧 [T, C, H, W]
        frames_tensor = torch.stack(frame_tensors, dim=0)

        # 创建SlowFast双路径输入
        slow_path = frames_tensor[::4].unsqueeze(0)  # [1, 8, C, H, W]
        fast_path = frames_tensor.unsqueeze(0)  # [1, 32, C, H, W]

        # 调整维度 [B, C, T, H, W]
        slow_path = slow_path.transpose(1, 2).contiguous()  # [1, C, 8, H, W]
        fast_path = fast_path.transpose(1, 2).contiguous()  # [1, C, 32, H, W]

        return slow_path, fast_path

    def predict_frames(self, frames: List[np.ndarray]) -> Dict:
        """对视频帧进行预测"""
        try:
            with torch.no_grad():
                # 预处理帧
                slow_input, fast_input = self.preprocess_frames(frames)
                slow_input = slow_input.to(self.device)
                fast_input = fast_input.to(self.device)

                # 模型预测
                outputs = self.model([slow_input, fast_input])
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                # 应用置信度阈值
                if confidence < self.confidence_threshold:
                    predicted_class = -1
                    class_name = "None"
                    status = "Low Confidence"
                else:
                    class_name = self.class_names.get(predicted_class, f'class_{predicted_class}') if hasattr(self,
                                                                                                              'class_names') else f'class_{predicted_class}'
                    status = "Confident"

                # 获取top-k预测结果
                top_k = min(3, self.num_classes)
                top_probs, top_indices = torch.topk(probabilities, top_k)

                top_predictions = []
                for i in range(top_k):
                    idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()

                    if prob < self.confidence_threshold:
                        pred_class_name = "None"
                        pred_status = "Low Confidence"
                    else:
                        pred_class_name = self.class_names.get(idx, f'class_{idx}') if hasattr(self,
                                                                                               'class_names') else f'class_{idx}'
                        pred_status = "Confident"

                    top_predictions.append({
                        'class_id': idx,
                        'class_name': pred_class_name,
                        'confidence': prob,
                        'status': pred_status,
                        'above_threshold': prob >= self.confidence_threshold
                    })

                result = {
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_name': class_name,
                    'status': status,
                    'above_threshold': confidence >= self.confidence_threshold,
                    'top_predictions': top_predictions,
                    'timestamp': time.time(),
                    'confidence_threshold': self.confidence_threshold
                }

                return result

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

    def _capture_frames(self, rtsp_url: str):
        """捕获RTSP流帧的线程函数 - 优化版本"""
        cap = None
        max_retries = 5
        retry_count = 0
        retry_delay = 2

        while self.is_running and retry_count < max_retries:
            try:
                # 尝试不同的RTSP URL格式
                rtsp_variants = [
                    rtsp_url,
                    rtsp_url.replace("?cmip=", ":554/live"),
                    rtsp_url + ":554/live",
                    rtsp_url.replace("rtsp://", "rtsp://admin:kemande1206@") + "/live"
                ]

                for rtsp_variant in rtsp_variants:
                    try:
                        print(f"尝试连接: {rtsp_variant}")
                        cap = cv2.VideoCapture(rtsp_variant)

                        # 设置OpenCV参数以提高性能
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                        cap.set(cv2.CAP_PROP_FPS, 15)

                        if cap.isOpened():
                            # 测试读取一帧
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                print(f"✅ 成功连接到RTSP流: {rtsp_variant}")

                                # 获取视频参数
                                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = int(cap.get(cv2.CAP_PROP_FPS))

                                if fps <= 0:
                                    fps = 15
                                if self.frame_width <= 0 or self.frame_height <= 0:
                                    self.frame_width = 640
                                    self.frame_height = 480

                                self.fps = fps
                                print(f"📹 视频参数: {self.frame_width}x{self.frame_height}, {fps}FPS")
                                break
                            else:
                                cap.release()
                                cap = None
                        else:
                            cap.release()
                            cap = None
                    except Exception as e:
                        if cap:
                            cap.release()
                        cap = None
                        continue

                if cap is None or not cap.isOpened():
                    raise ConnectionError("所有RTSP连接尝试都失败")

                logger.info(f"视频参数: {self.frame_width}x{self.frame_height}, {self.fps}FPS")

                last_frame_time = time.time()
                frame_interval = 1.0 / self.fps
                consecutive_errors = 0

                while self.is_running:
                    current_time = time.time()

                    # 控制帧率
                    if current_time - last_frame_time < frame_interval:
                        time.sleep(0.001)
                        continue

                    ret, frame = cap.read()
                    if ret and frame is not None:
                        consecutive_errors = 0  # 重置错误计数

                        # 更新帧率统计
                        self.frame_count += 1
                        if current_time - self.last_fps_time >= 1.0:
                            self.current_fps = self.frame_count
                            self.frame_count = 0
                            self.last_fps_time = current_time

                        # 转换BGR到RGB用于预测
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_buffer.append(frame_rgb)

                        # 控制预测频率，避免过度计算
                        current_time = time.time()
                        if (current_time - self.last_prediction_time >= self.prediction_interval and
                                len(self.frame_buffer) >= self.num_frames and
                                not self.prediction_queue.full()):
                            try:
                                frames_list = list(self.frame_buffer)[-self.num_frames:]
                                self.prediction_queue.put(frames_list, block=False)
                                self.last_prediction_time = current_time
                            except queue.Full:
                                pass

                        # 视频保存 - 关键修改：确保队列有数据
                        if self.save_video:
                            # 源视频帧
                            try:
                                self.original_video_queue.put(frame.copy(), block=False, timeout=0.01)
                            except queue.Full:
                                # 队列满时丢弃最老的帧
                                try:
                                    self.original_video_queue.get_nowait()
                                    self.original_video_queue.put(frame.copy(), block=False, timeout=0.01)
                                except:
                                    pass

                            # 标注视频帧
                            try:
                                annotated_frame = self._annotate_frame(frame.copy())
                                self.annotated_video_queue.put(annotated_frame, block=False, timeout=0.01)
                            except queue.Full:
                                try:
                                    self.annotated_video_queue.get_nowait()
                                    self.annotated_video_queue.put(annotated_frame, block=False, timeout=0.01)
                                except:
                                    pass

                        last_frame_time = current_time
                    else:
                        consecutive_errors += 1
                        logger.warning(f"读取帧失败 ({consecutive_errors}/5)")

                        if consecutive_errors >= 5:
                            logger.error("连续读取失败，尝试重新连接...")
                            break

                        time.sleep(0.1)

                # 正常退出内层循环
                break

            except Exception as e:
                logger.error(f"RTSP连接错误: {e}")
                retry_count += 1
                if cap:
                    cap.release()
                    cap = None

                if retry_count < max_retries:
                    logger.info(f"{retry_delay}秒后重试... ({retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避

        if cap:
            cap.release()
        logger.info("RTSP流捕获线程结束")

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """在帧上添加标注信息"""
        annotated_frame = frame.copy()

        # 添加时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 添加预测结果
        if self.latest_result and self.latest_result.get('success', False):
            result = self.latest_result
            class_name = result['class_name']
            confidence = result['confidence']
            status = result['status']

            # 根据状态选择颜色
            color = (0, 255, 0) if status == "Confident" else (0, 165, 255)

            # 添加分类和置信度
            pred_text = f"{class_name}: {confidence:.3f}"
            cv2.putText(annotated_frame, pred_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 添加录制状态和性能信息
        if self.current_original_writer is not None:
            recording_text = "REC"
            cv2.putText(annotated_frame, recording_text, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(annotated_frame, fps_text, (frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    def _start_new_video_segment(self):
        """开始新的视频片段"""
        if self.current_original_writer is not None:
            self._stop_current_video_segment()

        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_segment_count += 1

        # 源视频文件名
        original_filename = f"original_{timestamp}_part{self.video_segment_count:03d}.mp4"
        original_filepath = os.path.join(self.video_save_dir, "original", original_filename)

        # 标注视频文件名
        annotated_filename = f"annotated_{timestamp}_part{self.video_segment_count:03d}.mp4"
        annotated_filepath = os.path.join(self.video_save_dir, "annotated", annotated_filename)

        print(f"🎥 创建视频文件: {original_filepath}")
        print(f"🎥 创建标注文件: {annotated_filepath}")

        # 尝试多种编码器
        fourcc_options = ['avc1', 'mp4v', 'X264', 'MJPG']

        self.current_original_writer = None
        self.current_annotated_writer = None

        for fourcc_code in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                original_writer = cv2.VideoWriter(original_filepath, fourcc, self.fps,
                                                  (self.frame_width, self.frame_height))
                annotated_writer = cv2.VideoWriter(annotated_filepath, fourcc, self.fps,
                                                   (self.frame_width, self.frame_height))

                if original_writer.isOpened() and annotated_writer.isOpened():
                    self.current_original_writer = original_writer
                    self.current_annotated_writer = annotated_writer
                    print(f"✅ 使用编码器: {fourcc_code}")
                    break
                else:
                    original_writer.release()
                    annotated_writer.release()
            except Exception as e:
                print(f"编码器 {fourcc_code} 失败: {e}")
                continue

        if self.current_original_writer is None:
            print("❌ 所有编码器都失败，无法创建视频文件")
            return

        self.video_start_time = time.time()

        logger.info(f"开始录制视频片段 {self.video_segment_count}")
        print(f"🎥 开始录制视频片段 {self.video_segment_count}")

    def _stop_current_video_segment(self):
        """停止当前视频片段"""
        if self.current_original_writer is not None:
            try:
                self.current_original_writer.release()
                self.current_annotated_writer.release()
                duration = time.time() - self.video_start_time
                logger.info(f"停止录制视频片段 {self.video_segment_count}, 时长: {duration:.1f}秒")
                print(f"📹 停止录制视频片段 {self.video_segment_count}, 时长: {duration:.1f}秒")
            except Exception as e:
                print(f"释放视频写入器时出错: {e}")

            self.current_original_writer = None
            self.current_annotated_writer = None
            self.video_start_time = None

    def _original_video_worker(self):
        """源视频保存工作线程"""
        print("🎬 源视频保存线程启动")
        frames_processed = 0

        while self.is_running:
            try:
                # 获取源视频帧，设置较短超时
                frame = self.original_video_queue.get(timeout=0.1)
                frames_processed += 1

                # 检查是否需要开始新视频
                if self.current_original_writer is None:
                    print("🔄 启动第一个视频片段")
                    self._start_new_video_segment()
                elif time.time() - self.video_start_time >= self.video_duration:
                    print("🔄 达到时长限制，开始新片段")
                    self._start_new_video_segment()

                # 写入源视频帧
                if self.current_original_writer is not None:
                    self.current_original_writer.write(frame)

                    # 每处理100帧打印一次状态
                    if frames_processed % 100 == 0:
                        print(f"📊 源视频已处理 {frames_processed} 帧")

                self.original_video_queue.task_done()

            except queue.Empty:
                # 打印队列状态
                if frames_processed > 0 and frames_processed % 500 == 0:
                    print(f"⏳ 源视频队列空，已处理 {frames_processed} 帧")
                continue
            except Exception as e:
                logger.error(f"源视频保存错误: {e}")
                print(f"❌ 源视频保存错误: {e}")

    def _annotated_video_worker(self):
        """标注视频保存工作线程"""
        print("🎬 标注视频保存线程启动")
        frames_processed = 0

        while self.is_running:
            try:
                # 获取标注视频帧
                frame = self.annotated_video_queue.get(timeout=0.1)
                frames_processed += 1

                # 写入标注视频帧
                if self.current_annotated_writer is not None:
                    self.current_annotated_writer.write(frame)

                    # 每处理100帧打印一次状态
                    if frames_processed % 100 == 0:
                        print(f"📊 标注视频已处理 {frames_processed} 帧")

                self.annotated_video_queue.task_done()

            except queue.Empty:
                # 打印队列状态
                if frames_processed > 0 and frames_processed % 500 == 0:
                    print(f"⏳ 标注视频队列空，已处理 {frames_processed} 帧")
                continue
            except Exception as e:
                logger.error(f"标注视频保存错误: {e}")
                print(f"❌ 标注视频保存错误: {e}")

    def _prediction_worker(self):
        """预测工作线程"""
        while self.is_running:
            try:
                # 获取预测任务
                frames_list = self.prediction_queue.get(timeout=0.1)
                try:
                    result = self.predict_frames(frames_list)
                    self.latest_result = result
                    self.prediction_count += 1

                    # 输出预测结果
                    if result['success']:
                        self._print_prediction_result(result)

                except Exception as e:
                    logger.warning(f"实时预测失败: {e}")
                finally:
                    self.prediction_queue.task_done()
            except queue.Empty:
                continue

    def _print_prediction_result(self, result: Dict):
        """打印预测结果"""
        timestamp = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
        recording_indicator = "🔴" if self.current_original_writer is not None else "⚫"
        segment_info = f" Seg{self.video_segment_count}" if self.video_segment_count > 0 else ""

        print(
            f"[{timestamp}] {recording_indicator}{segment_info} {result['class_name']}: {result['confidence']:.3f} ({result['status']})")

    def start_realtime_prediction(self, rtsp_url: str):
        """开始实时预测"""
        if self.is_running:
            logger.warning("实时预测已经在运行")
            return

        self.is_running = True
        self.last_prediction_time = time.time()

        # 启动线程
        threads = [
            threading.Thread(target=self._capture_frames, args=(rtsp_url,), name="CaptureThread"),
            threading.Thread(target=self._prediction_worker, name="PredictionThread"),
            threading.Thread(target=self._original_video_worker, name="OriginalVideoThread"),
            threading.Thread(target=self._annotated_video_worker, name="AnnotatedVideoThread")
        ]

        for thread in threads:
            thread.daemon = True
            thread.start()

        # 保存线程引用
        self.capture_thread, self.prediction_thread, self.original_video_thread, self.annotated_video_thread = threads

        logger.info("开始实时预测")
        print("=" * 60)
        print("🚀 实时预测已启动")
        if self.save_video:
            print(f"📁 保存目录: {os.path.abspath(self.video_save_dir)}")
            print(f"⏱️  分段时长: {self.video_duration}秒")
        print(f"🎯 置信度阈值: {self.confidence_threshold}")
        print(f"⏰ 预测间隔: {self.prediction_interval}秒")
        print("⏹️  按 Ctrl+C 停止")
        print("=" * 60)

    def stop_realtime_prediction(self):
        """停止实时预测"""
        self.is_running = False

        if self.current_original_writer is not None:
            self._stop_current_video_segment()

        logger.info("停止实时预测")
        print("\n🛑 实时预测已停止")
        print(f"📊 总预测次数: {self.prediction_count}")
        if self.save_video:
            print(f"🎬 总视频片段: {self.video_segment_count}")

    def set_class_names(self, class_names: Dict[int, str]):
        """设置类别名称映射"""
        self.class_names = class_names

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"🎯 置信度阈值已更新: {self.confidence_threshold}")

    def set_prediction_interval(self, interval: float):
        """设置预测间隔"""
        self.prediction_interval = max(0.1, interval)
        print(f"⏰ 预测间隔已更新: {self.prediction_interval}秒")


class SlowFastModel(nn.Module):
    """SlowFast模型定义"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        try:
            self.model = slowfast_r50(pretrained=pretrained)
            in_features = self.model.blocks[6].proj.in_features
            self.model.blocks[6].proj = nn.Linear(in_features, num_classes)
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise e

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    try:
        model_path = r"/home/wuhao@kemande.cn/work/02_Video_Classification/checkpoints_shop/SlowFast/loss/best_loss_22.pth"

        if not os.path.exists(model_path):
            logger.warning("模型文件不存在，使用预训练模型")
            model_path = None

        predictor = SlowFastPredictor(
            model_path=model_path or "",
            num_classes=4,
            device="auto",
            confidence_threshold=0.6,
            save_video=True,
            video_save_dir="/home/wuhao@kemande.cn/work/02_Video_Classification/saved_videos",
            video_duration=300
        )

        predictor.set_class_names({
            0: "drinking", 1: "eating", 2: "tearing", 3: "hiding"
        })

        # 使用更可靠的RTSP URL格式
        rtsp_url = "rtsp://admin:kemande1206@192.168.121.248:554/live"  # 添加端口和路径
        predictor.start_realtime_prediction(rtsp_url)

        try:
            while predictor.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            predictor.stop_realtime_prediction()

    except Exception as e:
        logger.error(f"运行失败: {e}")
        print(f"错误: {e}")