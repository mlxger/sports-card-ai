# Sports Card Scope AI

<p align="center"><img src="assets/image.png" alt="CardScope AI 工作流程"></p>

<p align="center"><strong>多模态收藏卡检索、智能排序与结构化信息提取</strong></p>

<p align="center"><a href="README.md">英文</a> · <a href="#快速开始">快速开始</a> · <a href="#api-端点覆盖范围">API</a></p>

<p align="center"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"> <img src="https://img.shields.io/badge/FastAPI-Production%20API-009688?logo=fastapi&logoColor=white"> <img src="https://img.shields.io/badge/Milvus-Million--Scale%20ANN-00A1EA"> <img src="https://img.shields.io/badge/YOLOv8-Card%20Vision-111F68"> <img src="https://img.shields.io/badge/Qwen3--VL--8B-OCR%20%26%20Parsing-7B61FF"></p>

## 概述

CardScope AI 是一个面向体育卡和收藏卡的端到端智能平台。用户只需上传一张图片，系统便会依次完成卡片定位、几何校正、多模态特征提取、向量召回、学习排序、视觉语言解析和知识增强型结构化输出。

该平台针对反光、倾斜拍摄、复杂背景、中英文混排、外观高度相似的平行版本、序列编号、评级标签以及碎片化的交易市场元数据等场景而构建。

**图像 → 归一化 → 检索 → 重排序 → 理解 → 结构化 → 服务化**

## 功能特性

- 使用 YOLOv8 完成目标检测/分割、裁剪、透视校正和方向归一化。
- 融合 InsightFace buffalo_s、DINOv2 facebook/dinov2-with-registers-base、SLIP ConvNeXt convnext_base_w 和 OpenCLIP ViT-H-14。
- 通过 Milvus HNSW 进行 ANN 检索，并支持球员、交易市场和状态筛选。
- 使用 LightGBM 实现学习排序，综合利用嵌入向量、HOG/轮廓、边框和元数据信号。
- 通过 Ollama 调用 Qwen3-VL-8B，对单面和双面卡片进行 OCR。
- 基于 ChromaDB + BGE-M3 实现语义检索、字段检索和多条件 RAG 检索。
- 提供 FastAPI API、依赖诊断、并发控制、Docker，以及基于 Streamlit 的运维界面。
- 提供 Qwen3-VL LoRA 训练/评估和 LightGBM 排序模型训练流程。

## 核心能力

### 多模态检索

系统融合四路互补的视觉特征，生成 2,304 维搜索表示。Milvus HNSW 在保留产品元数据和筛选条件的同时，快速返回候选结果。

### 智能排序

LightGBM 综合视觉、轮廓、边框和元数据特征，将原始相似度转换为质量更高的最终排序。该排序模型既可离线训练，也可在 API 中启用。

### 卡片理解

Qwen3-VL 将卡片正反面图像解析为稳定的数据模式，包括球员、球队、品牌、系列、卡号、平行版本/变体、序列编号、位置、年份和评级字段。

### 知识增强型结构化处理

ChromaDB 存储卡片描述和元数据，支持语义检索、字段查询、Excel 导入及 RAG 辅助纠错。通过 Streamlit 查看器可以直观检查整个收藏库。

## 架构

~~~mermaid
flowchart LR
    A[卡片图像] --> B[YOLOv8 归一化]
    B --> C[InsightFace + DINOv2 + SLIP + OpenCLIP]
    C --> D[Milvus HNSW 召回]
    D --> E[LightGBM 重排序]
    B --> F[Qwen3-VL OCR]
    F --> G[ChromaDB + BGE-M3 RAG]
    E --> H[已排序产品]
    G --> I[结构化卡片 JSON]
~~~

## 环境要求

Python 3.10–3.13、Git、8 GB 以上内存；高吞吐量嵌入向量生成和 LoRA 训练需要 CUDA；向量检索需要 Milvus；Qwen3-VL OCR 需要 Ollama。

## 安装

~~~bash
git clone https://github.com/mlxger/sports-card-scope-ai.git
cd sports-card-scope-ai
python -m venv .venv
pip install -e ".[dev,retrieval,preprocessing,ranking,parsing]"
copy .env.example .env
~~~

## 快速开始

~~~bash
card-pipeline-api
uvicorn router.api:app --host 0.0.0.0 --port 8000
card-pipeline-doctor
docker compose up -d
card-pipeline-create-collection
card-pipeline-index-images data/cards --batch-size 100
~~~

打开 http://localhost:8000/docs 即可使用交互式 OpenAPI 控制台。

## 使用示例

### 搜索

~~~bash
curl -X POST http://localhost:8000/api/v1/retrieval/search -F "image=@card.jpg" -F "top_k=5" -F "rerank=true"
~~~

### YOLO 分割与归一化

~~~dotenv
CARD_PIPELINE_PREPROCESSING_MODE=yolo
CARD_PIPELINE_YOLO_MODEL_PATH=models/detection/yolov8_card.pt
~~~

在建立索引和执行检索时，系统会自动应用 YOLO。

### OCR

受资源限制，目前尚未提供经过 LoRA 微调的 Qwen3-VL 模型。本仓库的原生项目使用 Ollama 提供的 Qwen3-VL-8B 模型，该模型在实际测试中的准确率超过 90%。
下方链接提供了一部分 OCR 训练图像数据，你可以根据本仓库的模型配置自行训练模型。

~~~bash
ollama serve
ollama pull qwen3-vl:8b-instruct-q8_0
curl -X POST http://localhost:8000/api/v1/ocr/recognize/single -F "image=@card.jpg" -F 'fields=["name","brand","series","card_number"]'
curl -X POST http://localhost:8000/api/v1/ocr/recognize/double -F "front=@front.jpg" -F "back=@back.jpg"
~~~

### RAG 与查看器

~~~bash
curl http://localhost:8000/api/v1/rag/fields
curl http://localhost:8000/api/v1/rag/count
curl -X POST http://localhost:8000/api/v1/rag/search -H "Content-Type: application/json" -d '{"query":"Stephen Curry 2024 Olympic Games","top_k":5}'
card-pipeline-chroma-viewer
streamlit run src/knowledge/chroma_viewer.py
~~~

### OCR 训练与评估

~~~bash
python -m ocr_trainer.prepare_dataset data/annotations.jsonl data/llamafactory
python -m ocr_trainer.train data/llamafactory Qwen/Qwen3-VL-8B-Instruct outputs/card-ocr-lora
python -m ocr_trainer.predict data/eval.jsonl Qwen/Qwen3-VL-8B-Instruct data/predictions.jsonl --adapter outputs/card-ocr-lora
python -m ocr_trainer.evaluate data/eval.jsonl data/predictions.jsonl
~~~

### LightGBM 排序模型

~~~bash
card-pipeline-train-ranker data/ranking.csv models/ranking/ranking_model.joblib --enable-env .env
~~~

### Milvus 向量数据管理

API 还提供供数据摄取服务使用的集合生命周期管理和向量实体操作。嵌入向量数组的维度必须与配置的 2,304 维多模态数据模式一致。

~~~bash
# 创建 HNSW 集合和标量索引
curl -X POST http://localhost:8000/api/v1/milvus/collections/create

# 插入单条向量实体
curl -X POST http://localhost:8000/api/v1/milvus/records \
  -H "Content-Type: application/json" \
  -d '{"image_id":"card-001","tool_id":"marketplace-a","player_id":"player-001","status":0,"embedding":[0.01,0.02]}'

# 批量插入向量实体
curl -X POST http://localhost:8000/api/v1/milvus/records/batch \
  -H "Content-Type: application/json" \
  -d '{"records":[{"image_id":"card-001","tool_id":"marketplace-a","player_id":"player-001","status":0,"embedding":[0.01,0.02]}]}'

# 查看数量、获取自动生成的主键或删除记录
curl http://localhost:8000/api/v1/milvus/count
curl http://localhost:8000/api/v1/milvus/records/123
curl -X DELETE http://localhost:8000/api/v1/milvus/records \
  -H "Content-Type: application/json" -d '{"primary_keys":[123,124]}'
~~~

实际请求时，请发送由多模态编码器生成的完整 2,304 维嵌入向量。仍可通过 `card-pipeline-index-images` 完成从图像到向量的索引构建。

## API 端点覆盖范围

| 模块 | 端点 |
| --- | --- |
| 健康检查 | GET /health |
| 检索 | POST /api/v1/retrieval/search |
| Milvus | POST /api/v1/milvus/collections/create; GET /api/v1/milvus/count; POST /api/v1/milvus/records; POST /api/v1/milvus/records/batch; GET /api/v1/milvus/records/{primary_key}; DELETE /api/v1/milvus/records |
| OCR | GET /api/v1/ocr/fields; POST /api/v1/ocr/recognize/single; POST /api/v1/ocr/recognize/double |
| RAG | GET /api/v1/rag/fields; GET /api/v1/rag/count; 卡片 CRUD；批量导入；语义/字段/多条件检索及 Excel 导入 |
| 诊断 | GET /api/v1/system/dependencies |

## 项目结构

~~~text
src/{preprocessing,models,retrieval,rerank,ocr_parsing,knowledge,service,router}
scripts/
ocr_trainer/
configs/
tests/
assets/
~~~

## 模型与数据工作流

~~~text
models/detection/yolov8_card.pt
models/ranking/ranking_model.joblib
models/rag/BAAI/bge-m3/
data/cards/
data/ocr/{train,validation,test}.*
~~~

`.env.example` 文件集中管理设备、Milvus、Ollama、ChromaDB、嵌入向量缓存和模型路径配置。

## 模型与数据链接

此链接提供：基于 YOLOv8 训练的卡片图像裁剪模型、包含 20 多万张球员收藏卡的数据集，以及适用于 OCR 后训练的人工标注体育收藏卡图像数据：[链接](https://pan.baidu.com/s/1XUsRj9SJlq7E2hbFE3CWuw)

由于这些数据可能涉及法律风险，请通过 fengyanlin128@gmail.com 联系获取提取码。

⚠️ 版权声明

本项目仅发布算法代码和标注元数据（JSON 标签）。

如需将图像数据集用于学术研究或个人用途，请通过电子邮件联系项目作者申请获取。

该数据集仅限个人学术研究使用，禁止用于商业用途、二次分发或公开传播原始卡片图像。

## 评估

~~~bash
pytest
~~~

OCR 评估器会报告记录准确率、字段微平均/宏平均准确率、各字段准确率，以及字段存在性判断的精确率/召回率/F1。检索实验根据图库标签计算 Recall@K 和 MRR@K。

## Docker

~~~bash
docker build -t cardscope-ai .
docker run --rm -p 8000:8000 --env-file .env -v ./models:/app/models -v ./data:/app/data cardscope-ai
~~~

## 许可证

参见 LICENSE。重新分发前，请检查 Qwen、InsightFace、DINOv2、OpenCLIP/SLIP、YOLOv8、BGE-M3、Milvus、ChromaDB 及项目数据集各自的许可证。

## 致谢

Ultralytics、InsightFace、Meta DINOv2、LAION OpenCLIP/SLIP、LightGBM、Milvus、ChromaDB、Ollama、Qwen 和 LLaMA Factory。

## 免责声明

CardScope AI 是一套工程与研究工具。对于高价值卡片、真伪鉴定、定价和库存决策，应人工复核相似度结果和提取字段。运营方需自行负责数据权利、模型许可证、安全和部署策略。
