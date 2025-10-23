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

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

        # 预处理变换 - 与训练时保持一致
        self.transform = self._get_transform()

        logger.info(f"SlowFast预测器初始化完成，使用设备: {self.device}")

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        try:
            # 创建模型结构 - 与训练时相同
            model = SlowFastModel(self.num_classes, pretrained=False)

            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)

            # 处理多GPU训练保存的权重
            state_dict = self._process_checkpoint(checkpoint)

            # 加载权重
            model.load_state_dict(state_dict)
            model.to(self.device)

            logger.info("模型加载成功")
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
        """
        预处理视频帧 - 与训练时的collate_fn保持一致
        """
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

        # 创建SlowFast双路径输入 - 与训练时完全一致
        slow_path = frames_tensor[::4].unsqueeze(0)  # [1, 8, C, H, W]
        fast_path = frames_tensor.unsqueeze(0)  # [1, 32, C, H, W]

        # 调整维度 [B, C, T, H, W] - 与训练时完全一致
        slow_path = slow_path.transpose(1, 2).contiguous()  # [1, C, 8, H, W]
        fast_path = fast_path.transpose(1, 2).contiguous()  # [1, C, 32, H, W]

        return slow_path, fast_path

    def predict_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        对视频帧进行预测
        """
        try:
            with torch.no_grad():
                # 预处理帧
                slow_input, fast_input = self.preprocess_frames(frames)
                slow_input = slow_input.to(self.device)
                fast_input = fast_input.to(self.device)

                # 模型预测 - 使用列表输入，与训练时一致
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
                    class_name = self.class_names.get(idx, f'class_{idx}') if hasattr(self,
                                                                                      'class_names') else f'class_{idx}'
                    top_predictions.append({
                        'class_id': idx,
                        'class_name': class_name,
                        'confidence': prob
                    })

                result = {
                    'success': True,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_name': self.class_names.get(predicted_class, f'class_{predicted_class}') if hasattr(self,
                                                                                                               'class_names') else f'class_{predicted_class}',
                    'all_probabilities': all_probs.tolist(),
                    'top_predictions': top_predictions
                }

                return result

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict_video_file(self, video_path: str) -> Dict:
        """
        对视频文件进行预测
        """
        try:
            if not os.path.exists(video_path):
                return {
                    'success': False,
                    'error': f"视频文件不存在: {video_path}"
                }

            # 使用decord读取视频抽帧
            vr = decord.VideoReader(video_path, num_threads=4)
            total_frames = len(vr)

            if total_frames == 0:
                return {
                    'success': False,
                    'error': "视频文件没有帧"
                }

            # 读取所有帧
            frames = []
            # frames = vr.get_batch(range(len(vr))).asnumpy()
            for i in range(total_frames):
                try:
                    # 将VideoFrame对象转换为numpy数组格式
                    frame = vr[i].asnumpy()
                    frames.append(frame)
                except Exception as e:
                    logger.warning(f"读取第{i}帧失败: {e}")
                    continue

            # 如果视频太长，限制帧数以节省内存，下采样
            if len(frames) > self.num_frames * 4:
                # 均匀分布的采样索引
                indices = np.linspace(0, len(frames) - 1, self.num_frames * 2, dtype=int)
                # 下采样
                frames = [frames[i] for i in indices]

            # 进行预测
            result = self.predict_frames(frames)
            result['total_frames'] = total_frames
            result['processed_frames'] = len(frames)

            return result

        except Exception as e:
            logger.error(f"视频预测失败: {e}")
            return {
                'success': False,
                'error': f"视频处理失败: {str(e)}"
            }

    def set_class_names(self, class_names: Dict[int, str]):
        """设置类别名称映射"""
        self.class_names = class_names

    def update_class_names_in_result(self, result: Dict) -> Dict:
        """更新结果中的类别名称"""
        if hasattr(self, 'class_names') and result.get('success', False):
            predicted_id = result.get('predicted_class', 0)
            result['class_name'] = self.class_names.get(predicted_id, f'class_{predicted_id}')

            for pred in result.get('top_predictions', []):
                class_id = pred.get('class_id', 0)
                pred['class_name'] = self.class_names.get(class_id, f'class_{class_id}')

        return result


class SlowFastModel(nn.Module):
    """SlowFast模型定义 - 与训练时完全一致"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        try:
            # 加载预训练的SlowFast R50模型
            self.model = slowfast_r50(pretrained=pretrained)

            # 替换分类头以适应你的类别数
            in_features = self.model.blocks[6].proj.in_features
            self.model.blocks[6].proj = nn.Linear(in_features, num_classes)

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise e

    def forward(self, x):
        return self.model(x)


# 使用示例和测试代码
def test_predictor():
    """测试预测器"""
    try:
        # 尝试加载模型
        model_path = "/home/wuhao@kemande.cn/work/02_Video_Classification/checkpoints_shop/SlowFast/acc/best_acc_51.pth"

        # 检查模型文件
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            logger.info("使用预训练模型进行测试...")
            model_path = None

        if model_path and os.path.exists(model_path):
            predictor = SlowFastPredictor(
                model_path=model_path,
                num_classes=4,
                device="auto"
            )
        else:
            # 使用预训练模型
            predictor = SlowFastPredictor(
                model_path="",  # 空路径会触发预训练模型
                num_classes=4,
                device="auto"
            )

        # 设置类别名称
        class_names = {
            0: "drinking",
            1: "eating",
            2: "tearing",
            3: "hiding"
        }
        predictor.set_class_names(class_names)

        # 测试视频文件
        test_video = "/mnt/Data/daihongliang/shop2-clean/drinking/10月10日 (1)(1)312314214-1.mp4"
        if os.path.exists(test_video):
            logger.info(f"测试视频: {test_video}")
            result = predictor.predict_video_file(test_video)
            result = predictor.update_class_names_in_result(result)

            if result['success']:
                print(f"预测成功!")
                print(f"类别: {result['class_name']}")
                print(f"置信度: {result['confidence']:.4f}")
                print(f"总帧数: {result['total_frames']}")
                print("Top预测:")
                for pred in result['top_predictions']:
                    print(f"  {pred['class_name']}: {pred['confidence']:.4f}")
            else:
                print(f"预测失败: {result['error']}")
        else:
            logger.warning(f"测试视频不存在: {test_video}")
            print("请提供有效的测试视频路径")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误详情: {e}")


if __name__ == "__main__":
    test_predictor()