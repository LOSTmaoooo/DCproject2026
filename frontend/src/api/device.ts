import request from './request'
import type { Device, AlarmRecord, DashboardStats } from '@/types'

/**
 * 获取所有设备列表（用于下拉筛选）
 * GET /api/devices
 */
export function getDeviceList() {
  return request.get<unknown, Device[]>('/devices')
}

/**
 * 获取告警列表（未读优先）
 * GET /api/alarms
 */
export function getAlarmList(params?: { page?: number; pageSize?: number; isRead?: boolean }) {
  return request.get<unknown, AlarmRecord[]>('/alarms', { params })
}

/**
 * 获取统计概览数据
 * GET /api/dashboard/stats
 */
export function getDashboardStats() {
  return request.get<unknown, DashboardStats>('/dashboard/stats')
}
