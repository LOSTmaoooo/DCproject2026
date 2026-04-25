import type { FaultSeverity, FaultStatus, FaultType, DeviceStatus } from '@/enums'

// ============================================================
// 统一响应体结构
// ============================================================
export interface ApiResponse<T = unknown> {
  code: number       // 业务状态码，200 表示成功
  message: string    // 提示信息
  data: T            // 业务数据
}

// ============================================================
// 分页请求参数
// ============================================================
export interface PageQuery {
  page: number       // 当前页，从 1 开始
  pageSize: number   // 每页条数，默认 20
}

// ============================================================
// 分页响应结构
// ============================================================
export interface PageResult<T> {
  list: T[]          // 数据列表
  total: number      // 总条数
  page: number       // 当前页
  pageSize: number   // 每页条数
}

// ============================================================
// 设备信息
// ============================================================
export interface Device {
  deviceId: string         // 设备唯一ID
  deviceName: string       // 设备名称
  deviceCode: string       // 设备编号
  location: string         // 设备位置/车间
  status: DeviceStatus     // 设备状态
  lastOnlineAt: string     // 最后在线时间，ISO8601
}

// ============================================================
// 故障记录（列表用轻量结构）
// ============================================================
export interface FaultRecord {
  faultId: string              // 故障唯一ID
  deviceId: string             // 所属设备ID
  deviceName: string           // 所属设备名称
  faultType: FaultType | string // 故障类型
  severity: FaultSeverity      // 严重等级 1-4
  status: FaultStatus          // 处理状态
  description: string          // 故障描述
  detectedAt: string           // 检测时间，ISO8601，例：2024-03-15T14:30:00+08:00
  resolvedAt: string | null    // 解决时间，未解决时为 null
  imageUrl: string             // 故障图片URL
  heatmapUrl: string           // 热力图URL（MuSc输出）
  score: number                // 异常评分 0.0-1.0，越大越异常
}

// ============================================================
// 故障详情（包含完整分析结果）
// ============================================================
export interface FaultDetail extends FaultRecord {
  anomalyRegions: AnomalyRegion[]  // 异常区域列表
  analysisResult: AnalysisResult   // 模型分析详情
  handler: string | null           // 处理人
  handlerNote: string | null       // 处理备注
  createdAt: string                // 记录创建时间
  updatedAt: string                // 记录更新时间
}

// ============================================================
// 异常区域（坐标框）
// ============================================================
export interface AnomalyRegion {
  x: number        // 左上角 x 坐标（像素）
  y: number        // 左上角 y 坐标（像素）
  width: number    // 宽度（像素）
  height: number   // 高度（像素）
  score: number    // 该区域异常分数
  label: string    // 区域标签
}

// ============================================================
// 模型分析结果
// ============================================================
export interface AnalysisResult {
  modelVersion: string    // 模型版本
  inferenceTime: number   // 推理耗时（毫秒）
  threshold: number       // 判断阈值
  rawScore: number        // 原始异常分数
  clusterLabel: string    // AnomalyNCD 聚类标签
}

// ============================================================
// 故障列表查询参数
// ============================================================
export interface FaultQueryParams extends PageQuery {
  deviceId?: string           // 设备ID（可选）
  faultType?: string          // 故障类型（可选）
  severity?: FaultSeverity    // 严重等级（可选）
  status?: FaultStatus        // 处理状态（可选）
  startTime?: string          // 开始时间，ISO8601（可选）
  endTime?: string            // 结束时间，ISO8601（可选）
  keyword?: string            // 关键词搜索（可选）
}

// ============================================================
// 告警记录
// ============================================================
export interface AlarmRecord {
  alarmId: string          // 告警ID
  faultId: string          // 关联故障ID
  deviceId: string         // 设备ID
  deviceName: string       // 设备名称
  alarmLevel: FaultSeverity // 告警等级
  alarmMsg: string         // 告警内容
  isRead: boolean          // 是否已读
  createdAt: string        // 告警时间
}

// ============================================================
// 统计概览
// ============================================================
export interface DashboardStats {
  totalFaults: number      // 总故障数
  pendingFaults: number    // 待处理故障数
  todayFaults: number      // 今日新增故障数
  resolvedRate: number     // 解决率 0.0-1.0
  deviceTotal: number      // 设备总数
  deviceOnline: number     // 在线设备数
}
