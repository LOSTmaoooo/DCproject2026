import request from './request'
import type {
  FaultRecord,
  FaultDetail,
  FaultQueryParams,
  PageResult
} from '@/types'

/**
 * 获取故障列表（分页+筛选）
 * GET /api/faults
 */
export function getFaultList(params: FaultQueryParams) {
  return request.get<unknown, PageResult<FaultRecord>>('/faults', { params })
}

/**
 * 获取故障详情
 * GET /api/faults/:faultId
 */
export function getFaultDetail(faultId: string) {
  return request.get<unknown, FaultDetail>(`/faults/${faultId}`)
}

/**
 * 更新故障状态（处理/忽略）
 * PATCH /api/faults/:faultId/status
 */
export function updateFaultStatus(faultId: string, payload: { status: number; handlerNote?: string }) {
  return request.patch<unknown, FaultRecord>(`/faults/${faultId}/status`, payload)
}
