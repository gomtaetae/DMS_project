import { createRouter, createWebHistory } from "vue-router";
import Dashboard from "../views/Dashboard.vue";
import Tables from "../views/Tables.vue";
import Profile from "../views/Profile.vue";
import Signup from "../views/Register.vue";
import Signin from "../views/Login.vue";
import Restarea from "../examples/Table/RestArea.vue"

const routes = [
  {
    path: "/",
    name: "Dashboard",
    component: Dashboard,
  },
  {
    path: "/tables",
    name: "Tables",
    component: Tables,
  },
  {
    path: "/profile",
    name: "Profile",
    component: Profile,
  },
  {
    path: "/signin",
    name: "Signin",
    component: Signin,
  },
  {
    path: "/signup",
    name: "Signup",
    component: Signup,
  },
  {
    path: "/restarea",
    name: "RestArea",
    component: Restarea,
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
  linkActiveClass: "active",
});

router.beforeEach((to, from, next) => {
  // 보호된 경로를 배열로 정의
  const protectedRoutes = ['Profile']; // 'Profile' 경로를 보호 경로로 설정

  if (protectedRoutes.includes(to.name) && !localStorage.getItem('token')) {
    next({ name: 'Signin' });
  } else {
    next();
  }
});


export default router;
