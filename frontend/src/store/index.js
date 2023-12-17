import { createStore } from "vuex";
import axios from 'axios';

export default createStore({
  state: {
    hideConfigButton: false,
    isPinned: true,
    showConfig: false,
    sidebarType: "bg-white",
    isRTL: false,
    mcolor: "",
    darkMode: false,
    isNavFixed: false,
    isAbsolute: false,
    showNavs: true,
    showSidenav: true,
    showNavbar: true,
    showFooter: true,
    showMain: true,
    layout: "default",
    token: null,
    isLoggedIn: false,
    usersId: null,
    nickname: '',
    usersName: '',
    email: '',
    usersBirth: null,
    usersPhone: '',
    addressMain: '',
    addressSub: '',

  },
  mutations: {
    toggleConfigurator(state) {
      state.showConfig = !state.showConfig;
    },
    navbarMinimize(state) {
      const sidenav_show = document.querySelector(".g-sidenav-show");

      if (sidenav_show.classList.contains("g-sidenav-hidden")) {
        sidenav_show.classList.remove("g-sidenav-hidden");
        sidenav_show.classList.add("g-sidenav-pinned");
        state.isPinned = true;
      } else {
        sidenav_show.classList.add("g-sidenav-hidden");
        sidenav_show.classList.remove("g-sidenav-pinned");
        state.isPinned = false;
      }
    },
    sidebarType(state, payload) {
      state.sidebarType = payload;
    },
    navbarFixed(state) {
      if (state.isNavFixed === false) {
        state.isNavFixed = true;
      } else {
        state.isNavFixed = false;
      }
    },
    SET_LOGIN_STATUS(state, status) {
      state.isLoggedIn = status;
    },
    SET_TOKEN(state, token) {
      state.token = token;
      state.isLoggedIn = !!token; // 토큰의 존재 여부에 따라 로그인 상태 변경
    },
    SET_USER_DETAILS(state, userDetails) {
      state.usersId = userDetails.usersId;
      state.nickname = userDetails.nickname;
      state.usersName = userDetails.usersName;
      state.email = userDetails.email;
      state.usersBirth = userDetails.usersBirth;
      state.usersPhone = userDetails.usersPhone;
      state.addressMain = userDetails.addressMain;
      state.addressSub = userDetails.addressSub;
    },
  },
  actions: {
    toggleSidebarColor({ commit }, payload) {
      commit("sidebarType", payload);
    },
    login({ commit }, token) {
      localStorage.setItem('token', token); // 토큰을 로컬 스토리지에 저장
      commit('SET_TOKEN', token); // 토큰을 스토어에 저장하고 로그인 상태를 변경
      console.log('로그인 토큰:', token); // 로그인 토큰 확인
    },
    logout({ commit }) {
      localStorage.removeItem('token'); // 토큰을 로컬 스토리지에서 제거
      commit('SET_TOKEN', null); // 스토어의 토큰 제거 및 로그인 상태 변경
    },
    checkAuthentication({ commit }) {
      const token = localStorage.getItem('token');
      if (token) {
        // 여기에서 토큰의 유효성을 검증하는 로직을 추가할 수 있습니다.
        // 예를 들어, JWT 토큰이 만료되지 않았는지 확인할 수 있습니다.
        commit('SET_LOGIN_STATUS', true);
      } else {
        commit('SET_LOGIN_STATUS', false);
      }
    },
    fetchUserDetails({ commit }) {
      axios.get('/api/user/details')
        .then(response => {
          console.log('유저 정보:', response.data); // 사용자 정보 확인
          const userDetails = response.data;
          commit('SET_USER_DETAILS', userDetails);
        })
        .catch(error => {
          console.error("Error fetching user details:", error);
          // 오류 처리 로직 추가 (예: 오류 메시지 표시)
        });
    },
  },
  getters: {}
});
