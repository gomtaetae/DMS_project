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
                                    <h4 class="font-weight-bolder">Sign In</h4>
                                    <p class="mb-0">Enter your email and password to sign in</p>
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
                                        <argon-switch id="rememberMe" v-model="userInfo.remember">Remember me</argon-switch>

                                        <div class="text-center">
                                            <argon-button type="submit" class="mt-4" variant="gradient" color="success"
                                                fullWidth size="lg">Sign in</argon-button>
                                        </div>
                                    </form>
                                </div>
                                <div class="px-1 pt-0 text-center card-footer px-lg-2">
                                    <p class="mx-auto mb-4 text-sm">
                                        Don't have an account?
                                        <a href="javascript:;" class="text-success text-gradient font-weight-bold"
                                            @click="goToSignUp">Sign up</a>
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div
                            class="top-0 my-auto text-center col-6 d-lg-flex d-none h-100 pe-0 position-absolute end-0 justify-content-center flex-column">
                            <div class="position-relative bg-gradient-primary h-100 m-3 px-7 border-radius-lg d-flex flex-column justify-content-center overflow-hidden"
                                style="background-image: url('https://raw.githubusercontent.com/creativetimofficial/public-assets/master/argon-dashboard-pro/assets/img/signin-ill.jpg');
                                background-size: cover;">
                                <span class="mask bg-gradient-success opacity-6"></span>
                                <h4 class="mt-5 text-white font-weight-bolder position-relative">"Attention is the new
                                    currency"</h4>
                                <p class="text-white position-relative">The more effortless the writing looks, the more
                                    effort the writer actually put into the process.</p>
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
            }
        }
    },
    methods: {
        makeLogin() {
            let path = `http://${window.location.hostname}:5000/api/auth/signin`;
            console.log("Login request:", this.userInfo); // 로그인 요청 정보 출력

            axios.post(path, this.userInfo)
                .then((res) => {
                    console.log("Login response:", res); // 성공 응답 출력

                    if (res.data.success) {
                        // 성공적로직
                        this.$router.push("/");
                    } else {
                        // 로그인 실패 메시지를 출력하는 부분
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
        }
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