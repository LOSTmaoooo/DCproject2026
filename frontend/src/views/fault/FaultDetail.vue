<template>
  <div class="page-wrapper">
    <!-- 顶部 -->
    <header class="page-header">
      <div style="display:flex;align-items:center;gap:8px">
        <span class="back-btn" @click="router.back()">← 返回</span>
        <h1>故障详情</h1>
      </div>
      <div style="display:flex;gap:8px">
        <el-button
          v-if="detail?.status === FaultStatus.PENDING"
          type="primary"
          size="small"
          @click="openHandleDialog(FaultStatus.PROCESSING)"
        >开始处理</el-button>
        <el-button
          v-if="detail?.status === FaultStatus.PROCESSING"
          type="success"
          size="small"
          @click="openHandleDialog(FaultStatus.RESOLVED)"
        >标记已解决</el-button>
        <el-button
          v-if="detail?.status === FaultStatus.PENDING"
          size="small"
          @click="openHandleDialog(FaultStatus.IGNORED)"
        >忽略</el-button>
      </div>
    </header>

    <div class="page-content" v-loading="loading">
      <template v-if="detail">
        <!-- 基本信息 -->
        <div class="card detail-section">
          <h3>基本信息</h3>
          <el-descriptions :column="3" border>
            <el-descriptions-item label="故障ID">{{ detail.faultId }}</el-descriptions-item>
            <el-descriptions-item label="设备名称">{{ detail.deviceName }}</el-descriptions-item>
            <el-descriptions-item label="设备ID">{{ detail.deviceId }}</el-descriptions-item>
            <el-descriptions-item label="故障类型">{{ FaultTypeLabel[detail.faultType] ?? detail.faultType }}</el-descriptions-item>
            <el-descriptions-item label="严重等级">
              <el-tag :type="FaultSeverityTagType[detail.severity] as any" size="small">
                {{ FaultSeverityLabel[detail.severity] }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="处理状态">
              <el-tag :type="FaultStatusTagType[detail.status] as any" size="small">
                {{ FaultStatusLabel[detail.status] }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="异常评分">
              <el-progress
                :percentage="+(detail.score * 100).toFixed(1)"
                :color="scoreColor(detail.score)"
                :stroke-width="8"
                style="width:180px"
              />
            </el-descriptions-item>
            <el-descriptions-item label="检测时间">{{ formatDateTime(detail.detectedAt) }}</el-descriptions-item>
            <el-descriptions-item label="解决时间">{{ formatDateTime(detail.resolvedAt) }}</el-descriptions-item>
            <el-descriptions-item label="处理人" :span="1">{{ detail.handler ?? '暂无' }}</el-descriptions-item>
            <el-descriptions-item label="处理备注" :span="2">{{ detail.handlerNote ?? '暂无' }}</el-descriptions-item>
            <el-descriptions-item label="故障描述" :span="3">{{ detail.description }}</el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- 图像区域 -->
        <div class="card detail-section">
          <h3>故障图像</h3>
          <div class="image-group">
            <div class="image-box">
              <div class="img-label">原始图片</div>
              <img :src="detail.imageUrl" alt="原始图片" />
            </div>
            <div class="image-box">
              <div class="img-label">异常热力图（MuSc 输出）</div>
              <img :src="detail.heatmapUrl" alt="热力图" />
            </div>
          </div>
        </div>

        <!-- 模型分析 -->
        <div class="card detail-section">
          <h3>模型分析结果</h3>
          <el-descriptions :column="3" border>
            <el-descriptions-item label="模型版本">{{ detail.analysisResult.modelVersion }}</el-descriptions-item>
            <el-descriptions-item label="推理耗时">{{ detail.analysisResult.inferenceTime }} ms</el-descriptions-item>
            <el-descriptions-item label="判断阈值">{{ detail.analysisResult.threshold }}</el-descriptions-item>
            <el-descriptions-item label="原始异常分数">{{ detail.analysisResult.rawScore }}</el-descriptions-item>
            <el-descriptions-item label="聚类标签（AnomalyNCD）" :span="2">{{ detail.analysisResult.clusterLabel }}</el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- 异常区域 -->
        <div class="card detail-section" v-if="detail.anomalyRegions?.length">
          <h3>异常区域列表</h3>
          <el-table :data="detail.anomalyRegions" border size="small">
            <el-table-column label="标签" prop="label" width="120" />
            <el-table-column label="X" prop="x" width="80" />
            <el-table-column label="Y" prop="y" width="80" />
            <el-table-column label="宽度" prop="width" width="80" />
            <el-table-column label="高度" prop="height" width="80" />
            <el-table-column label="区域异常分数" prop="score">
              <template #default="{ row }">
                {{ (row.score * 100).toFixed(1) }}%
              </template>
            </el-table-column>
          </el-table>
        </div>
      </template>

      <!-- 空状态 -->
      <el-empty v-else-if="!loading" description="未找到故障记录" />
    </div>

    <!-- 处理状态对话框 -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="420px">
      <el-form :model="handleForm" label-width="80px">
        <el-form-item label="处理备注">
          <el-input
            v-model="handleForm.handlerNote"
            type="textarea"
            :rows="3"
            placeholder="可选填写处理说明"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitLoading" @click="submitHandle">确认</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getFaultDetail, updateFaultStatus } from '@/api/fault'
import type { FaultDetail } from '@/types'
import {
  FaultSeverity, FaultSeverityLabel, FaultSeverityTagType,
  FaultStatus, FaultStatusLabel, FaultStatusTagType,
  FaultTypeLabel
} from '@/enums'
import { formatDateTime } from '@/utils/format'

const route = useRoute()
const router = useRouter()

const loading = ref(false)
const detail = ref<FaultDetail | null>(null)

// 处理对话框
const dialogVisible = ref(false)
const submitLoading = ref(false)
const targetStatus = ref<FaultStatus>(FaultStatus.PROCESSING)
const handleForm = ref({ handlerNote: '' })
const dialogTitle = computed(() => {
  const map: Record<number, string> = {
    [FaultStatus.PROCESSING]: '开始处理',
    [FaultStatus.RESOLVED]: '标记已解决',
    [FaultStatus.IGNORED]: '忽略该故障'
  }
  return map[targetStatus.value] ?? '更新状态'
})

function scoreColor(score: number) {
  if (score >= 0.8) return '#f56c6c'
  if (score >= 0.5) return '#e6a23c'
  return '#67c23a'
}

async function loadDetail() {
  loading.value = true
  try {
    detail.value = await getFaultDetail(route.params.faultId as string)
  } finally {
    loading.value = false
  }
}

function openHandleDialog(status: FaultStatus) {
  targetStatus.value = status
  handleForm.value.handlerNote = ''
  dialogVisible.value = true
}

async function submitHandle() {
  if (!detail.value) return
  submitLoading.value = true
  try {
    await updateFaultStatus(detail.value.faultId, {
      status: targetStatus.value,
      handlerNote: handleForm.value.handlerNote || undefined
    })
    ElMessage.success('状态已更新')
    dialogVisible.value = false
    loadDetail()
  } finally {
    submitLoading.value = false
  }
}

onMounted(loadDetail)
</script>
