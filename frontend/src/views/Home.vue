<template>
  <dashboard />
</template>

<script>
import Dashboard from "@/views/Dashboard.vue"; // 경로는 실제 Dashboard 컴포넌트의 위치에 맞게 조정

export default {
  name: "Home",
  components: {
    Dashboard, // 컴포넌트 이름을 변경
  },
};
</script>