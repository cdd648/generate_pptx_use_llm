# generate_pptx_use_llm

使用 Gemini LLM 将信息图图片转换为高还原度可编辑 PPTX 文件。

## 原理

1. **文本分析** — Gemini 视觉模型识别图片中所有文字的位置、字号、颜色、对齐方式
2. **文字擦除** — Gemini 图片编辑模型去除原图上的文字，保留背景、图标等视觉元素
3. **验证重试** — 自动检测擦除后是否有残留文字，如有则自动重试（最多 2 轮）
4. **PPTX 合成** — 干净背景图作为全屏幻灯片背景，在文字原位叠加可编辑透明文本框

## 安装

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`，填入 Gemini API Key：

```bash
cp .env.example .env
# 编辑 .env 填入 GEMINI_API_KEY
```

## 使用

```bash
# 单张图片
python generate_pptx.py --image <图片路径> --output output.pptx

# 批量处理目录下所有图片
python generate_pptx.py --image-dir <图片目录> --output output.pptx

# 保留擦除文字后的干净背景图
python generate_pptx.py --image-dir <图片目录> --output output.pptx --keep-clean
```

### 示例

```bash
python generate_pptx.py --image-dir examples/俄罗斯地理和文化 --output 俄罗斯地理和文化.pptx
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--image` | 单张图片路径 |
| `--image-dir` | 图片目录（按文件名排序） |
| `--output` | 输出 PPTX 路径（默认 output.pptx） |
| `--model` | 文本分析模型（默认 gemini-3-pro-preview） |
| `--image-edit-model` | 图片编辑模型（默认 gemini-3-pro-image-preview） |
| `--no-text` | 不添加文本叠加层，仅图片背景 |
| `--keep-clean` | 保留擦除文字后的干净图片 |

## 文章

本项目属于[《儿子要做PPT，我只好继续写代码》](https://mp.weixin.qq.com/s/cpiS5AOK7h6aBAfDAHl5VA)的演示代码项目。

关注公众号获取更多内容:

**AI Native启示录**

<img src="images/qrcode.jpg" width="200" />