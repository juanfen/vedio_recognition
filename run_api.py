import uvicorn
import os

if __name__ == "__main__":
    # 设置环境变量
    os.environ["MODEL_PATH"] = "/home/wuhao@kemande.cn/work/02_Video_Classification/checkpoints_shop/SlowFast/acc/best_acc.pth"
    os.environ["NUM_CLASSES"] = "4"

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )