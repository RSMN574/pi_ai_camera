#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$HOME/ai_camera"
BUILD_DIR="$PROJECT_DIR/cpp/build"
EXECUTABLE="$BUILD_DIR/yolo_ort"

cd "$PROJECT_DIR"

echo "=================================="
echo " Raspberry Pi 5 YOLOv8n C++"
echo "=================================="

# 檢查模型
if [[ ! -f "models/yolov8n.onnx" ]]; then
    echo "[錯誤] 找不到 models/yolov8n.onnx"
    exit 1
fi

# 檢查 ONNX Runtime
ORT_DIR="$PROJECT_DIR/onnxruntime-linux-aarch64-1.18.0"

if [[ ! -f "$ORT_DIR/lib/libonnxruntime.so" ]]; then
    echo "[錯誤] 找不到 ONNX Runtime"
    exit 1
fi

# 自動編譯
if [[ ! -x "$EXECUTABLE" ]] || \
   [[ cpp/detector.cpp -nt "$EXECUTABLE" ]] || \
   [[ cpp/preview.cpp -nt "$EXECUTABLE" ]] || \
   [[ cpp/CMakeLists.txt -nt "$EXECUTABLE" ]]; then
    echo "[1/2] 編譯程式..."
    cmake -S cpp -B "$BUILD_DIR"
    cmake --build "$BUILD_DIR" -j2
else
    echo "[1/2] 使用現有編譯結果"
fi

# 設定 ONNX Runtime 動態函式庫
export LD_LIBRARY_PATH="$ORT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[2/2] 啟動 YOLOv8n..."
echo "按 Q 結束預覽"
echo

exec "$EXECUTABLE"
