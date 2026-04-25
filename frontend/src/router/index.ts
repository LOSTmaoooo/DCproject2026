import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/fault'
    },
    {
      path: '/fault',
      name: 'FaultList',
      component: () => import('@/views/fault/FaultList.vue'),
      meta: { title: '故障列表' }
    },
    {
      path: '/fault/:faultId',
      name: 'FaultDetail',
      component: () => import('@/views/fault/FaultDetail.vue'),
      meta: { title: '故障详情' }
    }
  ]
})

export default router
