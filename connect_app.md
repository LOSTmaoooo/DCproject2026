# 前后端对接说明

## 1. 统一入口

这个仓库当前有两个 Streamlit 入口，二者都通过同一套 HTTP API 访问后端，不再直接调用 `core.*`。

- 本地入口：`streamlit run app/main_app.py`
- 云端入口：`streamlit run cloud/streamlit_app.py`

两者共用：
- `cloud/api_client.py`
- 同一组后端接口
- 同一套统一响应格式

## 2. 后端基础地址

前端通过环境变量 `API_BASE_URL` 指定后端地址。

默认值：

```text
http://localhost:8000/api/v1
```

## 3. 统一响应格式

所有接口都应返回同一结构：

```json
{
  "code": 200,
  "msg": "请求成功",
  "data": {}
}
```

约定：
- `code = 200`：成功
- `code != 200`：失败
- `msg`：前端直接展示给用户的提示
- `data`：业务数据

## 4. 接口清单

### 4.1 模型列表

- 方法：`GET`
- 地址：`/api/v1/models/list`
- 用途：获取单图推理可选模型列表

返回示例：

```json
{
  "code": 200,
  "msg": "ok",
  "data": [
    { "label": "default", "value": "default" },
    { "label": "musc_v1", "value": "musc_v1" }
  ]
}
```

前端也兼容字符串列表：

```json
{
  "code": 200,
  "msg": "ok",
  "data": ["default", "musc_v1"]
}
```

### 4.2 模式一：部件注册

- 方法：`POST`
- 地址：`/api/v1/mode1/register`
- 内容类型：`multipart/form-data`

请求字段：

| 字段 | 是否必填 | 类型 | 说明 | 示例 |
|---|---|---|---|---|
| `part_name` | 是 | string | 部件名称 | `bearing_ring` |
| `images` | 是 | file[] | 部件图片，可多张 | `image1.jpg` |

请求示例：

```json
{
  "part_name": "bearing_ring"
}
```

响应 `data` 推荐字段：

| 字段 | 说明 |
|---|---|
| `partName` | 部件名称 |
| `fileCount` | 上传图片数量 |
| `savedDir` | 保存目录 |

### 4.3 模式二：单图推理

- 方法：`POST`
- 地址：`/api/v1/mode2/predict`
- 内容类型：`multipart/form-data`

请求字段：

| 字段 | 是否必填 | 类型 | 说明 | 示例 |
|---|---|---|---|---|
| `image` | 是 | file | 单张待检测图片 | `sample.jpg` |
| `model_name` | 否 | string | 指定模型名称 | `musc_v1` |

响应 `data` 推荐字段：

| 字段 | 说明 |
|---|---|
| `score` | 异常分数 |
| `clusterId` | 聚类结果 ID |
| `anomalyType` | 异常类别 |
| `latencyMs` | 推理耗时 |
| `modelName` | 实际使用模型 |
| `imageUrl` | 原图地址（可选） |
| `heatmapUrl` | 热力图地址（可选） |

### 4.4 模式三：批量分析

- 方法：`POST`
- 地址：`/api/v1/mode3/batch_analysis`
- 内容类型：`multipart/form-data`

请求字段：

| 字段 | 是否必填 | 类型 | 说明 | 示例 |
|---|---|---|---|---|
| `file` | 是 | file | ZIP 数据包 | `dataset.zip` |

响应 `data` 推荐字段：

| 字段 | 说明 |
|---|---|
| `status` | 任务状态 |
| `resultDir` | 结果目录 |
| `info` | 处理提示 |

## 5. 前端界面映射

### 模式一
- 输入：部件名称、图片文件
- 输出：注册结果、文件数量、保存目录

### 模式二
- 输入：模型名称、单张图片
- 输出：异常分数、聚类 ID、异常类别、推理耗时

### 模式三
- 输入：ZIP 数据包
- 输出：结果目录、任务状态、提示信息

## 6. 错误展示规则

前端直接把 `msg` 当作用户提示展示。

建议后端返回更清晰的错误信息，例如：
- `图片格式不正确`
- `未找到可用模型`
- `ZIP 文件解压失败`
- `服务端返回非 JSON 数据`

## 7. 启动方式

### 本地调试

```bash
streamlit run app/main_app.py
```

### 云端入口

```bash
streamlit run cloud/streamlit_app.py
```

### 后端地址示例

```bash
set API_BASE_URL=http://localhost:8000/api/v1
```

## 8. 说明

- `app/main_app.py` 现在只负责打开同一套前端界面入口。
- `cloud/api_client.py` 负责所有 HTTP 请求。
- 如果后端新增字段，前端可直接透传展示，不需要修改整体交互结构。