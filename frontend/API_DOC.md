# 前后端对接文档

> 项目：工业故障检测系统（DCProject 2026）
> 前端技术栈：Vue3 + Vite + TypeScript + ElementPlus + Pinia
> 文档版本：v1.0 · 2026-04-25
> **本文档供后端开发直接参考，无需阅读前端代码**

---

## 一、基础规范

### 1.1 接口 BaseURL

```
http://<host>:8000/api
```

开发环境：前端通过 Vite Proxy 将 `/api/*` 转发到 `http://localhost:8000`，后端无需处理跨域。

生产环境：Nginx 反向代理，后端保持 `/api` 前缀。

---

### 1.2 统一请求头

| Header | 必填 | 说明 |
|---|---|---|
| `Content-Type` | 是 | `application/json` |
| `Authorization` | 是（需登录时） | `Bearer <JWT_TOKEN>` |

---

### 1.3 统一响应体结构（所有接口必须遵守）

```json
{
  "code": 200,
  "message": "success",
  "data": { }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | int | 业务状态码，**200 = 成功**，其他见 §1.5 |
| `message` | string | 提示信息，失败时前端直接弹出给用户 |
| `data` | any | 实际业务数据，失败时可为 `null` |

> ⚠️ **前端统一判断 `code === 200` 为成功，不依赖 HTTP status code 做业务判断（但 HTTP 状态码仍需正确返回）。**

---

### 1.4 分页规范

所有列表接口的分页参数和响应格式统一如下：

**请求参数（Query String）：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `page` | int | 1 | 当前页码，**从 1 开始** |
| `pageSize` | int | 20 | 每页条数，最大 100 |

**分页响应体（`data` 字段）：**

```json
{
  "list": [],
  "total": 150,
  "page": 1,
  "pageSize": 20
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `list` | array | 当前页数据列表 |
| `total` | int | 总条数（前端据此计算总页数） |
| `page` | int | 当前页（原样返回） |
| `pageSize` | int | 每页条数（原样返回） |

---

### 1.5 错误码规范

| code | HTTP Status | 含义 | 前端展示 |
|---|---|---|---|
| 200 | 200 | 成功 | 正常展示数据 |
| 400 | 400 | 请求参数错误 | 弹出 `message` |
| 401 | 401 | 未授权/Token 失效 | 弹出"未授权，请重新登录" |
| 403 | 403 | 无权限 | 弹出"无权限访问" |
| 404 | 404 | 资源不存在 | 弹出"请求的资源不存在" |
| 500 | 500 | 服务器内部错误 | 弹出"服务器内部错误" |

---

### 1.6 时间字段规范

- 所有时间字段统一使用 **ISO 8601 格式，含时区偏移**：
  ```
  2024-03-15T14:30:00+08:00
  ```
- 时间字段为空（如未解决时 `resolvedAt`）返回 **`null`**，不返回空字符串。
- 前端查询时传递的时间参数格式同上（DatePicker 已配置 `value-format="YYYY-MM-DDTHH:mm:ssZ"`）。

---

## 二、接口清单

### 2.1 获取故障列表

```
GET /api/faults
```

**请求参数（Query String）：**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `page` | int | 是 | 页码，从 1 开始 |
| `pageSize` | int | 是 | 每页条数 |
| `deviceId` | string | 否 | 设备ID，精确匹配 |
| `faultType` | string | 否 | 故障类型，见 §三 枚举定义 |
| `severity` | int | 否 | 严重等级 1-4，见 §三 |
| `status` | int | 否 | 处理状态 0-3，见 §三 |
| `startTime` | string | 否 | 检测时间起始，ISO8601 |
| `endTime` | string | 否 | 检测时间截止，ISO8601 |
| `keyword` | string | 否 | 关键词，模糊匹配设备名/故障描述 |

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "list": [
      {
        "faultId": "F20240315001",         // 故障唯一ID（字符串）
        "deviceId": "DEV_LINE_A_001",      // 所属设备ID
        "deviceName": "流水线A-视觉检测站1", // 设备名称
        "faultType": "surface_crack",      // 故障类型（枚举字符串）
        "severity": 3,                     // 严重等级 1=低危 2=中危 3=高危 4=严重
        "status": 0,                       // 处理状态 0=待处理 1=处理中 2=已解决 3=已忽略
        "description": "零件表面发现裂纹，位于左侧边缘区域",
        "detectedAt": "2024-03-15T14:30:00+08:00", // 检测时间
        "resolvedAt": null,                         // 解决时间，未解决为 null
        "imageUrl": "http://example.com/images/F20240315001.jpg",  // 原始图片URL
        "heatmapUrl": "http://example.com/heatmaps/F20240315001.png", // 热力图URL
        "score": 0.87                      // 异常评分 0.0-1.0，越大越异常
      }
    ],
    "total": 150,
    "page": 1,
    "pageSize": 20
  }
}
```

---

### 2.2 获取故障详情

```
GET /api/faults/{faultId}
```

**Path 参数：**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `faultId` | string | 是 | 故障ID |

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "faultId": "F20240315001",
    "deviceId": "DEV_LINE_A_001",
    "deviceName": "流水线A-视觉检测站1",
    "faultType": "surface_crack",
    "severity": 3,
    "status": 0,
    "description": "零件表面发现裂纹，位于左侧边缘区域",
    "detectedAt": "2024-03-15T14:30:00+08:00",
    "resolvedAt": null,
    "imageUrl": "http://example.com/images/F20240315001.jpg",
    "heatmapUrl": "http://example.com/heatmaps/F20240315001.png",
    "score": 0.87,
    "handler": null,           // 处理人姓名，未处理为 null
    "handlerNote": null,       // 处理备注，未处理为 null
    "createdAt": "2024-03-15T14:30:01+08:00",  // 记录创建时间
    "updatedAt": "2024-03-15T14:30:01+08:00",  // 最后更新时间
    "anomalyRegions": [
      {
        "x": 120,        // 异常区域左上角 x（像素坐标）
        "y": 85,         // 异常区域左上角 y（像素坐标）
        "width": 60,     // 区域宽度（像素）
        "height": 45,    // 区域高度（像素）
        "score": 0.92,   // 该区域异常分数
        "label": "裂纹"   // 区域标签
      }
    ],
    "analysisResult": {
      "modelVersion": "MuSc-v2.1+AnomalyNCD-v1.3", // 模型版本
      "inferenceTime": 234,     // 推理耗时（毫秒）
      "threshold": 0.65,        // 判断阈值
      "rawScore": 0.87,         // 原始异常分数（同 score）
      "clusterLabel": "cluster_3_crack_type_B" // AnomalyNCD 聚类标签
    }
  }
}
```

---

### 2.3 更新故障处理状态

```
PATCH /api/faults/{faultId}/status
```

**Path 参数：**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `faultId` | string | 是 | 故障ID |

**请求体（JSON）：**

```json
{
  "status": 2,
  "handlerNote": "已安排维修人员更换零部件"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `status` | int | 是 | 新状态值，见 §三 |
| `handlerNote` | string | 否 | 处理备注 |

**响应示例：**（返回更新后的故障记录，结构同列表单条数据）

```json
{
  "code": 200,
  "message": "状态更新成功",
  "data": { /* 同故障列表单条结构 */ }
}
```

---

### 2.4 获取设备列表（用于下拉筛选）

```
GET /api/devices
```

无需分页参数，返回全量设备（数量通常 <1000）。

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "deviceId": "DEV_LINE_A_001",
      "deviceName": "流水线A-视觉检测站1",
      "deviceCode": "VIS-A-001",          // 设备编号
      "location": "一车间A流水线",
      "status": 1,                         // 设备状态 0=离线 1=在线 2=故障 3=维护中
      "lastOnlineAt": "2024-03-15T14:28:00+08:00"
    }
  ]
}
```

---

### 2.5 获取统计概览

```
GET /api/dashboard/stats
```

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "totalFaults": 1248,      // 历史总故障数
    "pendingFaults": 37,      // 当前待处理故障数
    "todayFaults": 12,        // 今日新增故障数
    "resolvedRate": 0.823,    // 历史解决率（0.0-1.0），前端展示为 82.3%
    "deviceTotal": 24,        // 总设备数
    "deviceOnline": 21        // 当前在线设备数
  }
}
```

---

### 2.6 获取告警列表

```
GET /api/alarms
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `page` | int | 否 | 页码，默认 1 |
| `pageSize` | int | 否 | 每页条数，默认 20 |
| `isRead` | bool | 否 | `true`=已读，`false`=未读，不传=全部 |

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "alarmId": "ALM_20240315_001",
      "faultId": "F20240315001",        // 关联的故障ID
      "deviceId": "DEV_LINE_A_001",
      "deviceName": "流水线A-视觉检测站1",
      "alarmLevel": 3,                  // 告警等级，同 FaultSeverity 1-4
      "alarmMsg": "检测到高危故障：表面裂纹，异常分数 87%",
      "isRead": false,                  // 是否已读
      "createdAt": "2024-03-15T14:30:05+08:00"
    }
  ]
}
```

---

## 三、枚举定义

### 3.1 故障严重等级（severity）

| 值 | 含义 | 前端颜色 |
|---|---|---|
| `1` | 低危 | 灰色（info） |
| `2` | 中危 | 黄色（warning） |
| `3` | 高危 | 红色（danger） |
| `4` | 严重 | 红色（danger） |

### 3.2 故障处理状态（status）

| 值 | 含义 | 前端颜色 |
|---|---|---|
| `0` | 待处理 | 红色 |
| `1` | 处理中 | 黄色 |
| `2` | 已解决 | 绿色 |
| `3` | 已忽略 | 灰色 |

### 3.3 故障类型（faultType）

| 字符串值 | 中文含义 |
|---|---|
| `surface_scratch` | 表面划痕 |
| `surface_crack` | 表面裂纹 |
| `deformation` | 形变 |
| `contamination` | 污染 |
| `missing_part` | 缺件 |
| `corrosion` | 腐蚀 |
| `unknown` | 未知异常 |

> 如果模型检测出新类型，后端直接返回字符串，前端会原样展示（不会报错）。

### 3.4 设备状态（deviceStatus）

| 值 | 含义 |
|---|---|
| `0` | 离线 |
| `1` | 在线 |
| `2` | 故障 |
| `3` | 维护中 |

---

## 四、联调注意事项（避免报错清单）

### ❌ 字段不能缺

以下字段缺失会导致前端渲染崩溃或逻辑错误：

| 接口 | 必返回字段 |
|---|---|
| 故障列表单条 | `faultId`、`score`、`severity`、`status`、`detectedAt`、`imageUrl`、`heatmapUrl` |
| 故障详情 | 列表字段 + `anomalyRegions`（空数组`[]`可以，不能缺）+ `analysisResult`（对象不能缺） |
| 分页响应 | `list`、`total`、`page`、`pageSize` 四个都不能缺 |
| 统计概览 | 六个字段全部必须存在 |

### ⚠️ 格式要注意

1. **`score` 必须是 `0.0-1.0` 的浮点数**，前端乘 100 再展示为百分比，后端不要直接返回百分比值。
2. **`resolvedAt`、`handler`、`handlerNote` 为空时必须返回 `null`**，不能返回 `""`（空字符串）或缺省该字段。
3. **时间格式必须含时区**：`2024-03-15T14:30:00+08:00`，不接受 `2024-03-15 14:30:00`（无 T 和时区的格式）。
4. **`anomalyRegions` 没有数据时返回空数组 `[]`**，不能返回 `null` 或缺省。
5. **`faultType` 是字符串枚举**（如 `"surface_crack"`），不是数字，后端数据库存字符串。
6. **`severity` 和 `status` 是整数**（如 `3`），不是字符串（不能返回 `"3"`）。
7. **`resolvedRate` 是 `0.0-1.0` 的小数**，前端展示时会 `×100`，不要传百分比。

### 🔧 分页从 1 开始

前端 `page` 从 `1` 开始传，后端请勿按 0-based 处理，否则第一页会丢数据。

### 🖼️ 图片 URL

`imageUrl` 和 `heatmapUrl` 需要是**可直接访问的完整 URL**（含协议和域名），或者是前端能通过相对路径访问的路径。建议后端统一返回绝对 URL。

---

## 五、前端目录说明（供后端参考）

```
frontend/src/
├── api/
│   ├── request.ts       # axios 实例，统一拦截器，baseURL = /api
│   ├── fault.ts         # 故障相关接口
│   └── device.ts        # 设备/统计/告警接口
├── types/
│   └── index.ts         # 所有接口的 TypeScript 类型定义
├── enums/
│   └── index.ts         # 枚举值及中文映射（对应 §三）
├── views/fault/
│   ├── FaultList.vue    # 故障列表页（含筛选、分页）
│   └── FaultDetail.vue  # 故障详情页（含状态更新）
└── utils/
    └── format.ts        # 时间格式化工具
```
