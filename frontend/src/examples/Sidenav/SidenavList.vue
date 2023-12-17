<template>
  <div class="collapse navbar-collapse w-auto h-auto h-100" id="sidenav-collapse-main">
    <ul class="navbar-nav">
      <li class="nav-item">
        <sidenav-item url="/" :class="getRoute() === 'Dashboard' ? 'active' : ''" navText="메인화면">
          <template v-slot:icon>
            <img class="sideimg" src="../../assets/png/car-emergency.png" alt="대시보드">
          </template>
        </sidenav-item>
      </li>
      <li class="nav-item">
        <sidenav-item url="/tables" :class="getRoute() === 'tables' ? 'active' : ''" navText="월간 위험 행동 통계">
          <template v-slot:icon>
            <img class="sideimg" src="../../assets/png/chart-areaspline.png" alt="월간 위험 행동 통계" />
          </template>
        </sidenav-item>
      </li>
      <li class="navbar-item">
        <sidenav-item url="/restarea" :class="getRoute() === 'restarea' ? 'active' : ''" navText="휴게소&졸음쉼터">
          <template v-slot:icon>
            <img class="sideimg" src="../../assets/png/coffee.png" alt="휴게소및졸음쉼터" />
          </template>
        </sidenav-item>
      </li>
      <li class="mt-3 nav-item">
        <h6 class="text-xm ps-4 text-uppercase font-weight-bolder opacity-6">
          개인 페이지
        </h6>
      </li>
      <li class="nav-item" v-if="isLoggedIn">
        <sidenav-item url="/profile" :class="getRoute() === 'profile' ? 'active' : ''" navText="프로필">
          <template v-slot:icon>
            <img class="sideimg" src="../../assets/png/account.png" alt="프로필" />
          </template>
        </sidenav-item>
      </li>
      <li class="nav-item" v-if="!isLoggedIn">
        <sidenav-item url="/signin" :class="getRoute() === 'signin' ? 'active' : ''" navText="로그인">
          <template v-slot:icon>
            <img class="sideimg" src="../../assets/png/login-variant.png" alt="로그인" />
          </template>
        </sidenav-item>
      </li>
      <li class="nav-item" v-if="!isLoggedIn">
        <sidenav-item url="/signup" :class="getRoute() === 'signup' ? 'active' : ''" navText="회원가입">
          <template v-slot:icon>
            <img class="sideimg" src="../../assets/png/account-plus.png" alt="회원가입">
          </template>
        </sidenav-item>
      </li>
    </ul>
  </div>
</template>
<script>
import { mapState } from 'vuex';
import SidenavItem from "./SidenavItem.vue";

export default {
  name: "SidenavList",
  props: {
    cardBg: String
  },
  data() {
    return {
      title: "Argon Dashboard 2",
      controls: "dashboardsExamples",
      isActive: "active"
    };
  },
  components: {
    SidenavItem,
  },
  computed: {
    ...mapState(['isLoggedIn']),
  },
  methods: {
    getRoute() {
      const routeArr = this.$route.path.split("/");
      return routeArr[1];
    }
  }
};
</script>

<style>
.sideimg {
  width: 18px;
  height: 18px;
  align-self: center;
}
</style>