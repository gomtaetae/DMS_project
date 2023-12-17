<template>
    <div class="container top-0 position-sticky z-index-sticky">
        <div class="row">
            <div class="col-12">
                <navbar isBlur="blur  border-radius-lg my-3 py-2 start-0 end-0 mx-4 shadow" v-bind:darkMode="true"
                    isBtn="bg-gradient-success" />
            </div>
        </div>
    </div>
    <main class="mt-0 main-content">
        <section>
            <div class="page-header min-vh-100">
                <div class="container">
                    <div class="row">
                        <div class="mx-auto col-xl-4 col-lg-5 col-md-7 d-flex flex-column mx-lg-0">
                            <div class="card card-plain">
                                <div class="pb-0 card-header text-start">
                                    <h4 class="font-weight-bolder">로그인</h4>
                                    <p class="mb-0">이메일과 비밀번호를 입력해 로그인하세요</p>
                                </div>
                                <div class="card-body">
                                    <form role="form" @submit.prevent="makeLogin">
                                        <div class="mb-3">
                                            <argon-input type="email" placeholder="Email" name="email" size="lg"
                                                v-model="userInfo.email" />
                                        </div>
                                        <div class="mb-3">
                                            <argon-input type="password" placeholder="Password" name="password" size="lg"
                                                v-model="userInfo.password" />
                                        </div>
                                        <argon-switch id="rememberMe" v-model="userInfo.remember">아이디 저장</argon-switch>

                                        <div class="text-center">
                                            <argon-button type="submit" class="mt-4" variant="gradient" color="success"
                                                fullWidth size="lg">로그인</argon-button>
                                        </div>
                                    </form>
                                </div>
                                <div class="px-1 pt-0 text-center card-footer px-lg-2">
                                    <p class="mx-auto mb-4 text-sm">
                                        아직 회원이 아니신가요?
                                        <a href="javascript:;" class="text-success text-gradient font-weight-bold"
                                            @click="goToSignUp">회원가입</a>
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div
                            class="top-0 my-auto text-center col-6 d-lg-flex d-none h-100 pe-0 position-absolute end-0 justify-content-center flex-column">
                            <div class="position-relative bg-gradient-primary h-100 m-3 px-7 border-radius-lg d-flex flex-column justify-content-center overflow-hidden"
                            :style="{ backgroundImage: 'url(' + require('@/assets/img/loginbg.png') + ')',  backgroundSize: 'contain',  backgroundRepeat: 'no-repeat' }">
                                <span class="mask bg-gradient-success opacity-6"></span><br><br><br><br><br><br><br><br><br>
                                <h3 class="mt-5 text-white font-weight-bolder position-relative">"졸음 운전은 막을 수 있습니다"</h3>
                                <h4 class="text-white position-relative">졸음 운전, 운전 중 핸드폰 사용은 타인의 목숨을 앗아 갈 수 있습니다.</h4>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>
</template>

<script>
import Navbar from "@/examples/PageLayout/Navbar.vue";
import ArgonInput from "@/components/ArgonInput.vue";
import ArgonSwitch from "@/components/ArgonSwitch.vue";
import ArgonButton from "@/components/ArgonButton.vue";
import axios from 'axios';
const body = document.getElementsByTagName("body")[0];

export default {
    name: 'signin',
    components: {
        Navbar,
        ArgonInput,
        ArgonSwitch,
        ArgonButton,
    },
    data() {
        return {
            userInfo: {
                email: '',
                password: ''
            },
            accessToken: '', // 토큰을 저장할 변수 추가
        }
    },
    methods: {
        makeLogin() {
            let path = `http://${window.location.hostname}:5000/api/auth/signin`;
            console.log("Login request:", this.userInfo); // 로그인 요청 정보 출력

            axios.post(path, this.userInfo)
                .then((res) => {
                    console.log("Server response:", res);
                    if (res.data.access_token) {
                        this.fetchUserDetails(res.data.access_token); // 사용자 정보 요청 함수 호출
                        // 토큰을 localStorage에 저장
                        localStorage.setItem('token', res.data.access_token);
                        console.log("Token stored:", localStorage.getItem('token')); // 토큰 저장 확인 로그     
                        console.log("Redirecting to Dashboard..."); // 리디렉션 로그


                        // Vuex 상태 업데이트
                        this.$store.dispatch('login', res.data.access_token);
                        // 메인 페이지로 리다이렉트
                        this.$router.push({ name: "Dashboard" });
                    } else {
                        console.error('Login failed', res.data);
                    }
                })
                .catch((error) => {
                    console.error("Login error:", error); // 오류 정보 출력

                    if (error.response) {
                        // 서버로부터 받은 오류 응답
                        console.error('Error response data:', error.response.data);
                    } else {
                        // 오류 응답이 없는 경우
                        console.error('Error message:', error.message);
                    }
                });
        },
        fetchUserDetails(token) {
            // 토큰을 사용하여 사용자 정보를 요청하는 로직
            axios.get('/api/user/details', { headers: { 'Authorization': `Bearer ${token}` } })
                .then(response => {
                    this.$store.commit('SET_USER_DETAILS', response.data); // Vuex 스토어 업데이트
                })
                .catch(error => {
                    console.error("Error fetching user details:", error);
                });
        },
        makelogout() {
            // localStorage에서 토큰 제거
            localStorage.removeItem('token');
            // Vuex 상태 업데이트
            this.$store.dispatch('logout');
            // 로그인 페이지로 리다이렉트
            this.$router.push('/signin');
        },
    },
    goToSignUp() {
        this.$router.push('/signup');
    },
    created() {
        this.$store.state.hideConfigButton = true;
        this.$store.state.showNavbar = false;
        this.$store.state.showSidenav = false;
        this.$store.state.showFooter = false;
        body.classList.remove("bg-gray-100");
    },
    beforeUnmount() {
        this.$store.state.hideConfigButton = false;
        this.$store.state.showNavbar = true;
        this.$store.state.showSidenav = true;
        this.$store.state.showFooter = true;
        body.classList.add("bg-gray-100");
    },
};
</script>