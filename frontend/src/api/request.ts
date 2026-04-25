import axios from 'axios'
import { ElMessage } from 'element-plus'
import type { ApiResponse } from '@/types'
import { ApiCode } from '@/enums'

const request = axios.create({
  baseURL: '/api',          // 通过 vite proxy 转发到后端
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// ──────────────────────────────────────────
// 请求拦截器：注入 Token
// ──────────────────────────────────────────
request.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// ──────────────────────────────────────────
// 响应拦截器：统一错误处理
// ──────────────────────────────────────────
request.interceptors.response.use(
  (response) => {
    const res = response.data as ApiResponse

    // 业务成功
    if (res.code === ApiCode.SUCCESS) {
      return res.data as any
    }

    // 业务失败：弹出错误提示
    ElMessage.error(res.message || '请求失败')
    return Promise.reject(new Error(res.message))
  },
  (error) => {
    const status = error.response?.status
    const msgMap: Record<number, string> = {
      400: '请求参数错误',
      401: '未授权，请重新登录',
      403: '无权限访问',
      404: '请求的资源不存在',
      500: '服务器内部错误'
    }
    ElMessage.error(msgMap[status] || error.message || '网络错误')
    return Promise.reject(error)
  }
)

export default request
