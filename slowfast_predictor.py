import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import decord
from pytorchvideo.models.hub import slowfast_r50
import torchvision.transforms as T


class SlowFastPredictor:
    """SlowFast视频分类预测器"""

    def __init__(self, model_path: str, num_classes: int = 4, device: str = "auto"):
        """
        初始化SlowFast预测器

        Args:
            model_path: 训练好的模型权重路径
            num_classes: 类别数量
            device: 运行设备 ('auto', 'cuda', 'cpu')
        """
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.num_frames = 32

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 预处理变换
        self.transform = self._get_transform()

        print(f"SlowFast预测器初始化完成，使用设备: {self.device}")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        # 创建模型结构
        model = SlowFastModel(self.num_classes, pretrained=False)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)

        # 处理多GPU训练保存的权重
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 移除module前缀（如果使用DataParallel训练）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(self.device)

        return model

    def _get_transform(self):
        """获取预处理变换"""
        return T.Compose([
            T.Resize(256),
            lambda x: x / 255.0,
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_frames(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预处理视频帧

        Args:
            frames: 视频帧列表，每个帧为HWC格式的numpy数组

        Returns:
            slow_path: 慢路径输入张量 [1, 3, 8, 224, 224]
            fast_path: 快路径输入张量 [1, 3, 32, 224, 224]
        """
        if len(frames) < self.num_frames:
            # 如果帧数不足，重复最后一帧
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
            # 确保帧是RGB格式
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)

                # 转换为tensor并调整维度
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                frame_tensor = self.transform(frame_tensor)
                frame_tensors.append(frame_tensor)

        if not frame_tensors:
            raise ValueError("没有有效的帧可用于处理")

        # 堆叠帧 [T, C, H, W]
        frames_tensor = torch.stack(frame_tensors, dim=0)

        # 创建SlowFast双路径输入
        slow_path = frames_tensor[::4].unsqueeze(0)  # [1, 8, C, H, W]
        fast_path = frames_tensor.unsqueeze(0)  # [1, 32, C, H, W]

        # 调整维度 [B, C, T, H, W]
        slow_path = slow_path.transpose(1, 2).contiguous()
        fast_path = fast_path.transpose(1, 2).contiguous()

        return slow_path, fast_path

    def predict_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        对视频帧进行预测

        Args:
            frames: 视频帧列表

        Returns:
            预测结果字典
        """
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

                # 获取所有类别的概率
                all_probs = probabilities[0].cpu().numpy()

                # 获取top-k预测结果
                top_k = min(3, self.num_classes)
                top_probs, top_indices = torch.topk(probabilities, top_k)

                top_predictions = []
                for i in range(top_k):
                    idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    top_predictions.append({
                        'class_id': idx,
                        'class_name': f'class_{idx}',
                        'confidence': prob
                    })

                return {
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_name': f'class_{predicted_class}',
                    'all_probabilities': all_probs.tolist(),
                    'top_predictions': top_predictions
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def predict_video_file(self, video_path: str) -> Dict:
        """
        对视频文件进行预测

        Args:
            video_path: 视频文件路径

        Returns:
            预测结果字典
        """
        try:
            # 使用decord读取视频
            vr = decord.VideoReader(video_path, num_threads=4)
            total_frames = len(vr)

            # 读取所有帧
            frames = []
            for i in range(total_frames):
                frame = vr[i].asnumpy()  # 转换为numpy数组
                frames.append(frame)

            # 进行预测
            result = self.predict_frames(frames)
            result['total_frames'] = total_frames
            result['processed_frames'] = min(total_frames, self.num_frames)

            return result

        except Exception as e:
            return {
                'success': False,
                'error': f"视频读取失败: {str(e)}"
            }

    def set_class_names(self, class_names: Dict[int, str]):
        """
        设置类别名称映射

        Args:
            class_names: 类别ID到名称的映射字典
        """
        self.class_names = class_names

    def update_class_names_in_result(self, result: Dict) -> Dict:
        """更新结果中的类别名称"""
        if hasattr(self, 'class_names') and result['success']:
            predicted_id = result['predicted_class']
            result['class_name'] = self.class_names.get(predicted_id, f'class_{predicted_id}')

            for pred in result['top_predictions']:
                class_id = pred['class_id']
                pred['class_name'] = self.class_names.get(class_id, f'class_{class_id}')

        return result


class SlowFastModel(nn.Module):
    """SlowFast模型定义（与训练代码保持一致）"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # 加载预训练的SlowFast R50模型
        self.model = slowfast_r50(pretrained=pretrained)

        # 替换分类头以适应你的类别数
        in_features = self.model.blocks[6].proj.in_features
        self.model.blocks[6].proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)