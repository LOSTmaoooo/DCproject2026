import dayjs from 'dayjs'

/** ISO8601 → 本地时间字符串，e.g. 2024-03-15 14:30:00 */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '--'
  return dayjs(iso).format('YYYY-MM-DD HH:mm:ss')
}

/** ISO8601 → 日期，e.g. 2024-03-15 */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '--'
  return dayjs(iso).format('YYYY-MM-DD')
}

/** 将 dayjs 对象或字符串转成后端要求的 ISO8601（含时区偏移） */
export function toISOString(date: Date | string | null): string | undefined {
  if (!date) return undefined
  return dayjs(date).format('YYYY-MM-DDTHH:mm:ssZ')
}

/** 异常分数转百分比文字 */
export function scoreToPercent(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}
