<template>
  <div class="page-wrapper">
    <!-- 顶部导航 -->
    <header class="page-header">
      <h1>🏭 工业故障检测系统</h1>
      <span style="color:#909399;font-size:13px;">后端对接 Demo · Vue3 + ElementPlus</span>
    </header>

    <div class="page-content">
      <!-- 统计概览 -->
      <div class="stat-row">
        <div class="stat-card">
          <div class="label">总故障数</div>
          <div class="value">{{ stats.totalFaults }}</div>
        </div>
        <div class="stat-card">
          <div class="label">待处理</div>
          <div class="value danger">{{ stats.pendingFaults }}</div>
        </div>
        <div class="stat-card">
          <div class="label">今日新增</div>
          <div class="value warning">{{ stats.todayFaults }}</div>
        </div>
        <div class="stat-card">
          <div class="label">解决率</div>
          <div class="value success">{{ (stats.resolvedRate * 100).toFixed(1) }}%</div>
        </div>
        <div class="stat-card">
          <div class="label">在线设备</div>
          <div class="value">{{ stats.deviceOnline }} / {{ stats.deviceTotal }}</div>
        </div>
      </div>

      <!-- 筛选区 -->
      <div class="card">
        <div class="filter-row">
          <el-select
            v-model="query.deviceId"
            placeholder="全部设备"
            clearable
            style="width:180px"
          >
            <el-option
              v-for="d in deviceList"
              :key="d.deviceId"
              :label="d.deviceName"
              :value="d.deviceId"
            />
          </el-select>

          <el-select
            v-model="query.faultType"
            placeholder="故障类型"
            clearable
            style="width:150px"
          >
            <el-option
              v-for="(label, val) in FaultTypeLabel"
              :key="val"
              :label="label"
              :value="val"
            />
          </el-select>

          <el-select
            v-model="query.severity"
            placeholder="严重等级"
            clearable
            style="width:130px"
          >
            <el-option label="低危" :value="FaultSeverity.LOW" />
            <el-option label="中危" :value="FaultSeverity.MEDIUM" />
            <el-option label="高危" :value="FaultSeverity.HIGH" />
            <el-option label="严重" :value="FaultSeverity.CRITICAL" />
          </el-select>

          <el-select
            v-model="query.status"
            placeholder="处理状态"
            clearable
            style="width:130px"
          >
            <el-option label="待处理" :value="FaultStatus.PENDING" />
            <el-option label="处理中" :value="FaultStatus.PROCESSING" />
            <el-option label="已解决" :value="FaultStatus.RESOLVED" />
            <el-option label="已忽略" :value="FaultStatus.IGNORED" />
          </el-select>

          <el-date-picker
            v-model="timeRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
            value-format="YYYY-MM-DDTHH:mm:ssZ"
            style="width:360px"
          />

          <el-input
            v-model="query.keyword"
            placeholder="关键词搜索"
            clearable
            style="width:180px"
          />

          <el-button type="primary" @click="onSearch">查询</el-button>
          <el-button @click="onReset">重置</el-button>
        </div>
      </div>

      <!-- 表格 -->
      <div class="card" style="padding:0">
        <el-table
          :data="tableData"
          v-loading="loading"
          stripe
          style="width:100%"
          @row-click="goDetail"
          row-class-name="cursor-row"
        >
          <el-table-column label="故障ID" prop="faultId" width="180" />
          <el-table-column label="设备名称" prop="deviceName" min-width="140" />
          <el-table-column label="故障类型" prop="faultType" width="120">
            <template #default="{ row }">
              {{ FaultTypeLabel[row.faultType] ?? row.faultType }}
            </template>
          </el-table-column>
          <el-table-column label="严重等级" width="100">
            <template #default="{ row }">
              <el-tag :type="FaultSeverityTagType[row.severity as FaultSeverity] as any" size="small">
                {{ FaultSeverityLabel[row.severity as FaultSeverity] }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="处理状态" width="100">
            <template #default="{ row }">
              <el-tag :type="FaultStatusTagType[row.status as FaultStatus] as any" size="small">
                {{ FaultStatusLabel[row.status as FaultStatus] }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="异常分数" prop="score" width="100">
            <template #default="{ row }">
              <el-progress
                :percentage="+(row.score * 100).toFixed(1)"
                :color="scoreColor(row.score)"
                :stroke-width="6"
                :show-text="false"
                style="width:60px;display:inline-block;vertical-align:middle;margin-right:6px"
              />
              {{ scoreToPercent(row.score) }}
            </template>
          </el-table-column>
          <el-table-column label="检测时间" prop="detectedAt" width="175">
            <template #default="{ row }">{{ formatDateTime(row.detectedAt) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="100" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click.stop="goDetail(row)">详情</el-button>
            </template>
          </el-table-column>
        </el-table>

        <!-- 分页 -->
        <div style="padding:14px 20px;display:flex;justify-content:flex-end">
          <el-pagination
            v-model:current-page="query.page"
            v-model:page-size="query.pageSize"
            :total="total"
            :page-sizes="[10, 20, 50, 100]"
            layout="total, sizes, prev, pager, next, jumper"
            background
            @change="loadList"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getFaultList } from '@/api/fault'
import { getDeviceList, getDashboardStats } from '@/api/device'
import type { FaultRecord, Device, DashboardStats, FaultQueryParams } from '@/types'
import {
  FaultSeverity, FaultSeverityLabel, FaultSeverityTagType,
  FaultStatus, FaultStatusLabel, FaultStatusTagType,
  FaultTypeLabel
} from '@/enums'
import { formatDateTime, scoreToPercent, toISOString } from '@/utils/format'

const router = useRouter()

// ── 状态 ──────────────────────────────────────────
const loading = ref(false)
const tableData = ref<FaultRecord[]>([])
const total = ref(0)
const deviceList = ref<Device[]>([])
const timeRange = ref<[string, string] | null>(null)

const stats = reactive<DashboardStats>({
  totalFaults: 0,
  pendingFaults: 0,
  todayFaults: 0,
  resolvedRate: 0,
  deviceTotal: 0,
  deviceOnline: 0
})

const query = reactive<FaultQueryParams>({
  page: 1,
  pageSize: 20,
  deviceId: undefined,
  faultType: undefined,
  severity: undefined,
  status: undefined,
  startTime: undefined,
  endTime: undefined,
  keyword: undefined
})

// ── 工具 ──────────────────────────────────────────
function scoreColor(score: number) {
  if (score >= 0.8) return '#f56c6c'
  if (score >= 0.5) return '#e6a23c'
  return '#67c23a'
}

// ── 数据加载 ──────────────────────────────────────
async function loadList() {
  if (timeRange.value) {
    query.startTime = timeRange.value[0]
    query.endTime = timeRange.value[1]
  } else {
    query.startTime = undefined
    query.endTime = undefined
  }
  loading.value = true
  try {
    const res = await getFaultList(query)
    tableData.value = res.list
    total.value = res.total
  } finally {
    loading.value = false
  }
}

async function loadDevices() {
  deviceList.value = await getDeviceList()
}

async function loadStats() {
  const s = await getDashboardStats()
  Object.assign(stats, s)
}

// ── 交互 ──────────────────────────────────────────
function onSearch() {
  query.page = 1
  loadList()
}

function onReset() {
  query.page = 1
  query.pageSize = 20
  query.deviceId = undefined
  query.faultType = undefined
  query.severity = undefined
  query.status = undefined
  query.keyword = undefined
  timeRange.value = null
  loadList()
}

function goDetail(row: FaultRecord) {
  router.push({ name: 'FaultDetail', params: { faultId: row.faultId } })
}

// ── 初始化 ────────────────────────────────────────
onMounted(() => {
  loadList()
  loadDevices()
  loadStats()
})
</script>

<style scoped>
:deep(.cursor-row) { cursor: pointer; }
</style>
