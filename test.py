import torch
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
import torch.nn as nn


# 定义C3D模型（与训练时相同的结构）
class C3D(nn.Module):
    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits


class VideoPredictor:
    def __init__(self, model_path, num_classes=101, class_names_file=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        # 加载模型
        self.model = C3D(num_classes)

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)

        # 处理多GPU训练保存的模型
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除module前缀（如果使用DataParallel训练）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 加载类别名称
        self.class_names = self.load_class_names(class_names_file)

    def load_class_names(self, class_names_file):
        """加载类别名称"""
        if class_names_file and os.path.exists(class_names_file):
            class_names = {}
            with open(class_names_file, 'r') as f:
                for line in f:
                    idx, name = line.strip().split(' ')
                    class_names[int(idx) - 1] = name
            return class_names
        else:
            # 如果没有类别文件，返回数字索引
            return {i: f"Class_{i}" for i in range(self.num_classes)}

    def preprocess_video(self, video_path, num_frames=16):
        """预处理视频，提取帧并进行预处理"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 读取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (112, 112))
            frame_normalized = frame_resized.astype(np.float32)
            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
            frames.append(frame_chw)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"无法从视频 {video_path} 中读取帧")

        # 采样策略：从视频中均匀采样num_frames帧
        total_frames = len(frames)
        if total_frames >= num_frames:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            # 如果帧数不足，重复最后一帧
            sampled_frames = frames
            while len(sampled_frames) < num_frames:
                sampled_frames.append(frames[-1])

        # 转换为tensor
        frames_tensor = torch.tensor(np.stack(sampled_frames), dtype=torch.float32)
        frames_tensor = frames_tensor.unsqueeze(0)  # 添加batch维度
        frames_tensor = frames_tensor.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)

        return frames_tensor

    def predict(self, video_path):
        """对单个视频进行预测"""
        try:
            # 预处理视频
            input_tensor = self.preprocess_video(video_path)
            input_tensor = input_tensor.to(self.device)

            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # 获取类别名称
            class_name = self.class_names.get(predicted_class, f"Class_{predicted_class}")

            # 获取top-3预测结果
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top3_results = []
            for i in range(3):
                idx = top3_indices[0][i].item()
                prob = top3_probs[0][i].item()
                name = self.class_names.get(idx, f"Class_{idx}")
                top3_results.append((name, prob))

            return {
                'predicted_class': predicted_class,
                'class_name': class_name,
                'confidence': confidence,
                'top3_predictions': top3_results,
                'all_probabilities': probabilities.cpu().numpy()[0]
            }

        except Exception as e:
            return {'error': str(e)}

    def predict_batch(self, video_dir):
        """批量预测目录中的所有视频"""
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(f'*{ext}'))
            video_files.extend(Path(video_dir).glob(f'*{ext.upper()}'))

        results = {}
        for video_file in video_files:
            print(f"处理视频: {video_file.name}")
            result = self.predict(str(video_file))
            results[video_file.name] = result

        return results

    def save_prediction_video(self, video_path, output_path):
        """保存带预测结果的视频"""
        try:
            # 先进行一次预测获取结果
            result = self.predict(video_path)
            if 'error' in result:
                print(f"预测错误: {result['error']}")
                return None

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 设置视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
            print(f"预测结果: {result['class_name']} (置信度: {result['confidence']:.4f})")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 在帧上绘制预测结果
                cv2.putText(frame, f"Prediction: {result['class_name']}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {result['confidence']:.4f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 绘制Top-3预测结果
                for i, (name, prob) in enumerate(result['top3_predictions']):
                    y_pos = 120 + i * 30
                    color = (0, 255, 0) if i == 0 else (255, 255, 0)
                    cv2.putText(frame, f"{i + 1}. {name}: {prob:.4f}", (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                # 写入帧
                out.write(frame)

                # 显示进度
                if frame_count % 50 == 0:
                    print(f"处理进度: {frame_count}/{total_frames} 帧")

            # 释放资源
            cap.release()
            out.release()

            print(f"带预测结果的视频已保存到: {output_path}")
            return result

        except Exception as e:
            print(f"保存预测视频错误: {str(e)}")
            return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='C3D视频分类预测')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--video_path', type=str, required=True, help='要预测的视频路径或视频目录')
    parser.add_argument('--class_names', type=str, default=None, help='类别名称文件路径')
    parser.add_argument('--num_classes', type=int, default=101, help='类别数量')
    parser.add_argument('--output_video', type=str, default=None, help='输出视频路径')

    args = parser.parse_args()

    # 初始化预测器
    predictor = VideoPredictor(
        model_path=args.model_path,
        num_classes=args.num_classes,
        class_names_file=args.class_names
    )

    # 判断输入是文件还是目录
    if os.path.isfile(args.video_path):
        if args.output_video:
            # 保存带预测结果的视频
            result = predictor.save_prediction_video(
                args.video_path,
                args.output_video
            )
        else:
            # 单个视频预测
            result = predictor.predict(args.video_path)

        if result and 'error' not in result:
            print(f"\n预测结果:")
            print(f"视频文件: {os.path.basename(args.video_path)}")
            print(f"预测类别: {result['class_name']} (ID: {result['predicted_class']})")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"\nTop-3预测:")
            for i, (name, prob) in enumerate(result['top3_predictions']):
                print(f"  {i + 1}. {name}: {prob:.4f}")
        elif result and 'error' in result:
            print(f"预测错误: {result['error']}")

    elif os.path.isdir(args.video_path):
        # 批量预测
        results = predictor.predict_batch(args.video_path)

        print(f"\n批量预测结果:")
        for video_name, result in results.items():
            if 'error' in result:
                print(f"{video_name}: 错误 - {result['error']}")
            else:
                print(f"{video_name}: {result['class_name']} (置信度: {result['confidence']:.4f})")

    else:
        print(f"错误: {args.video_path} 不是有效的文件或目录")


if __name__ == "__main__":
    # 如果没有命令行参数，可以使用以下代码直接测试
    # main()

    # 或者直接使用以下代码进行测试
    model_path = "./checkpoints/acc/best_acc.pth"  # 修改为你的模型路径
    video_path = "/mnt/Data/daihongliang/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03.avi"  # 修改为你的视频路径
    class_names_file = "/mnt/Data/daihongliang/ucfTrainTestlist/classInd.txt"  # 修改为你的类别文件路径
    output_video_path = "output_with_predictions.avi"  # 保存结果视频

    if os.path.exists(model_path):
        predictor = VideoPredictor(
            model_path=model_path,
            num_classes=101,
            class_names_file=class_names_file
        )

        # 方法1: 简单预测（只输出文本结果）
        print("=== 简单预测 ===")
        result = predictor.predict(video_path)

        if 'error' in result:
            print(f"预测错误: {result['error']}")
        else:
            print(f"\n预测结果:")
            print(f"视频文件: {os.path.basename(video_path)}")
            print(f"预测类别: {result['class_name']} (ID: {result['predicted_class']})")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"\nTop-3预测:")
            for i, (name, prob) in enumerate(result['top3_predictions']):
                print(f"  {i + 1}. {name}: {prob:.4f}")

        # 方法2: 保存带预测结果的视频
        print("\n=== 保存预测视频 ===")
        result = predictor.save_prediction_video(
            video_path,
            output_video_path
        )

    else:
        print(f"模型文件不存在: {model_path}")