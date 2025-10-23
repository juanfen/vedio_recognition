from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import cv2
import numpy as np
import tempfile
from typing import List, Optional
import json
import io
from slowfast_predictor import SlowFastPredictor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="SlowFast视频分类API",
    description="基于SlowFast模型的视频行为分类接口",
    version="1.0.0"
)

# 全局预测器实例
predictor = None


class PredictionRequest(BaseModel):
    """预测请求模型"""
    video_path: Optional[str] = None


class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool
    predicted_class: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    all_probabilities: Optional[List[float]] = None
    top_predictions: Optional[List[dict]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    """批量预测请求模型"""
    video_paths: List[str]


class BatchPredictionResponse(BaseModel):
    """批量预测响应模型"""
    results: List[dict]
    total: int
    success_count: int


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global predictor
    try:
        model_path = os.getenv("MODEL_PATH", "/home/wuhao@kemande.cn/work/02_Video_Classification/checkpoints_shop/SlowFast/acc/best_acc.pth")
        num_classes = int(os.getenv("NUM_CLASSES", "4"))

        predictor = SlowFastPredictor(
            model_path=model_path,
            num_classes=num_classes,
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

        logger.info("SlowFast预测器初始化成功")

    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        raise e


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "SlowFast视频分类API",
        "status": "运行中",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_upload": "/predict/upload",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="预测器未初始化")

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(predictor.device)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(request: PredictionRequest):
    """通过视频路径进行预测"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="预测器未初始化")

    if not request.video_path or not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail="视频路径不存在")

    try:
        import time
        start_time = time.time()

        # 进行预测
        result = predictor.predict_video_file(request.video_path)
        result = predictor.update_class_names_in_result(result)

        processing_time = time.time() - start_time
        result['processing_time'] = processing_time

        logger.info(f"预测完成: {request.video_path} -> {result.get('class_name', 'Unknown')}")

        return result

    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(file: UploadFile = File(...)):
    """通过文件上传进行预测"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="预测器未初始化")

    # 检查文件类型
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="请上传视频文件")

    try:
        import time
        start_time = time.time()

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            # 读取上传的文件内容
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # 进行预测
        result = predictor.predict_video_file(temp_path)
        result = predictor.update_class_names_in_result(result)

        processing_time = time.time() - start_time
        result['processing_time'] = processing_time

        # 清理临时文件
        os.unlink(temp_path)

        logger.info(f"上传预测完成: {file.filename} -> {result.get('class_name', 'Unknown')}")

        return result

    except Exception as e:
        logger.error(f"上传预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """批量预测多个视频"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="预测器未初始化")

    results = []
    success_count = 0

    for video_path in request.video_paths:
        try:
            if os.path.exists(video_path):
                result = predictor.predict_video_file(video_path)
                result = predictor.update_class_names_in_result(result)
                result['video_path'] = video_path
                results.append(result)

                if result['success']:
                    success_count += 1
            else:
                results.append({
                    'success': False,
                    'error': f"文件不存在: {video_path}",
                    'video_path': video_path
                })

        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'video_path': video_path
            })

    return BatchPredictionResponse(
        results=results,
        total=len(request.video_paths),
        success_count=success_count
    )


@app.get("/predict/realtime")
async def realtime_prediction():
    """实时预测端点（可用于流式处理）"""
    # 这里可以实现实时视频流预测
    return {"message": "实时预测端点", "status": "开发中"}


@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="预测器未初始化")

    return {
        "model_type": "SlowFast",
        "num_classes": predictor.num_classes,
        "num_frames": predictor.num_frames,
        "device": str(predictor.device),
        "class_names": getattr(predictor, 'class_names', {})
    }


@app.post("/predict/frames")
async def predict_from_frames(frames: List[UploadFile] = File(...)):
    """通过上传的帧序列进行预测"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="预测器未初始化")

    if len(frames) < predictor.num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"帧数量不足，至少需要 {predictor.num_frames} 帧，当前 {len(frames)} 帧"
        )

    try:
        import time
        start_time = time.time()

        # 读取所有帧
        frame_list = []
        for frame_file in frames:
            if not frame_file.content_type.startswith('image/'):
                continue

            content = await frame_file.read()
            nparr = np.frombuffer(content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame_rgb)

        if len(frame_list) < predictor.num_frames:
            raise HTTPException(status_code=400, detail="有效帧数量不足")

        # 进行预测
        result = predictor.predict_frames(frame_list)
        result = predictor.update_class_names_in_result(result)

        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['frames_processed'] = len(frame_list)

        return result

    except Exception as e:
        logger.error(f"帧预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"帧预测失败: {str(e)}")


# 错误处理
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "内部服务器错误"}
    )


@app.exception_handler(400)
async def bad_request_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"success": False, "error": str(exc.detail)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )