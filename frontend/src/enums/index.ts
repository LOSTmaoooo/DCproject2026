// ============================================================
// 故障严重等级
// ============================================================
export enum FaultSeverity {
  LOW = 1,      // 低危
  MEDIUM = 2,   // 中危
  HIGH = 3,     // 高危
  CRITICAL = 4  // 严重
}

export const FaultSeverityLabel: Record<FaultSeverity, string> = {
  [FaultSeverity.LOW]: '低危',
  [FaultSeverity.MEDIUM]: '中危',
  [FaultSeverity.HIGH]: '高危',
  [FaultSeverity.CRITICAL]: '严重'
}

export const FaultSeverityTagType: Record<FaultSeverity, string> = {
  [FaultSeverity.LOW]: 'info',
  [FaultSeverity.MEDIUM]: 'warning',
  [FaultSeverity.HIGH]: 'danger',
  [FaultSeverity.CRITICAL]: 'danger'
}

// ============================================================
// 故障处理状态
// ============================================================
export enum FaultStatus {
  PENDING = 0,    // 待处理
  PROCESSING = 1, // 处理中
  RESOLVED = 2,   // 已解决
  IGNORED = 3     // 已忽略
}

export const FaultStatusLabel: Record<FaultStatus, string> = {
  [FaultStatus.PENDING]: '待处理',
  [FaultStatus.PROCESSING]: '处理中',
  [FaultStatus.RESOLVED]: '已解决',
  [FaultStatus.IGNORED]: '已忽略'
}

export const FaultStatusTagType: Record<FaultStatus, string> = {
  [FaultStatus.PENDING]: 'danger',
  [FaultStatus.PROCESSING]: 'warning',
  [FaultStatus.RESOLVED]: 'success',
  [FaultStatus.IGNORED]: 'info'
}

// ============================================================
// 故障类型（对应模型检测出的异常类别）
// ============================================================
export enum FaultType {
  SURFACE_SCRATCH = 'surface_scratch',   // 表面划痕
  SURFACE_CRACK = 'surface_crack',       // 表面裂纹
  DEFORMATION = 'deformation',           // 形变
  CONTAMINATION = 'contamination',       // 污染
  MISSING_PART = 'missing_part',         // 缺件
  CORROSION = 'corrosion',               // 腐蚀
  UNKNOWN = 'unknown'                    // 未知异常
}

export const FaultTypeLabel: Record<string, string> = {
  [FaultType.SURFACE_SCRATCH]: '表面划痕',
  [FaultType.SURFACE_CRACK]: '表面裂纹',
  [FaultType.DEFORMATION]: '形变',
  [FaultType.CONTAMINATION]: '污染',
  [FaultType.MISSING_PART]: '缺件',
  [FaultType.CORROSION]: '腐蚀',
  [FaultType.UNKNOWN]: '未知异常'
}

// ============================================================
// 设备状态
// ============================================================
export enum DeviceStatus {
  ONLINE = 1,      // 在线
  OFFLINE = 0,     // 离线
  FAULT = 2,       // 故障
  MAINTENANCE = 3  // 维护中
}

export const DeviceStatusLabel: Record<DeviceStatus, string> = {
  [DeviceStatus.ONLINE]: '在线',
  [DeviceStatus.OFFLINE]: '离线',
  [DeviceStatus.FAULT]: '故障',
  [DeviceStatus.MAINTENANCE]: '维护中'
}

// ============================================================
// 统一 API 错误码
// ============================================================
export enum ApiCode {
  SUCCESS = 200,
  BAD_REQUEST = 400,
  UNAUTHORIZED = 401,
  FORBIDDEN = 403,
  NOT_FOUND = 404,
  SERVER_ERROR = 500
}
