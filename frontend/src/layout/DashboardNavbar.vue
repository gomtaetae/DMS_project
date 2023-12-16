<template>
    <base-nav class="navbar-top navbar-dark" id="navbar-main" :show-toggle-button="false" expand>
        <form class="navbar-search navbar-search-dark form-inline mr-3 d-none d-md-flex ml-lg-auto">
            <div class="form-group mb-0">
                <base-input placeholder="Search" class="input-group-alternative" alternative=""
                    addon-right-icon="fas fa-search">
                </base-input>
            </div>
        </form>
        <ul class="navbar-nav align-items-center d-none d-md-flex">
            <li class="nav-item dropdown">
                <base-dropdown class="nav-link pr-0">
                    <div class="media align-items-center" slot="title">
                        <span class="avatar avatar-sm rounded-circle">
                            <img alt="Image placeholder" src="img/theme/team-4-800x800.jpg">
                        </span>
                        <div class="media-body ml-2 d-none d-lg-block">
                            <span class="mb-0 text-sm font-weight-bold">{{ isAuthenticated ? users_name : 'Guest' }}</span>
                        </div>
                    </div>

                    <template>
                        <div class=" dropdown-header noti-title">
                            <h6 class="text-overflow m-0">Welcome!</h6>
                        </div>
                        <router-link to="/profile" class="dropdown-item">
                            <i class="ni ni-single-02"></i>
                            <span>My profile</span>
                        </router-link>
                        <router-link to="/profile" class="dropdown-item">
                            <i class="ni ni-settings-gear-65"></i>
                            <span>Settings</span>
                        </router-link>
                        <router-link to="/profile" class="dropdown-item">
                            <i class="ni ni-calendar-grid-58"></i>
                            <span>Activity</span>
                        </router-link>
                        <router-link to="/profile" class="dropdown-item">
                            <i class="ni ni-support-16"></i>
                            <span>Support</span>
                        </router-link>
                        <div class="dropdown-divider"></div>
                        <a v-if="isAuthenticated" @click="makelogout" class="dropdown-item">
                            <i class="ni ni-user-run"></i>
                            <span>Logout</span>
                        </a>
                        <router-link v-else to="/signin" class="dropdown-item">
                            <i class="ni ni-key-25"></i>
                            <span>Sign In</span>
                        </router-link>
                    </template>
                </base-dropdown>
            </li>
        </ul>
    </base-nav>
</template>
<script>
import isValidJwt from '../utils'

export default {
    data() {
        return {
            users_name: '',
            isAuthenticated: false,
            activeNotifications: false,
            showMenu: false,
            searchQuery: ''
        };
    },
    methods: {
        makelogout() {
            localStorage.removeItem('token');
            this.isAuthenticated = false; // 로그아웃 상태로 설정
            this.users_name = ''; // 사용자 이름 클리어
            this.$router.push('/signin'); // 사용자를 로그인 페이지로 리다이렉트합니다.
        },
        checkAuthentication() {
            const token = localStorage.getItem('token');
            console.log("Token from localStorage:", token); // 토큰 출력
            if (token) {
                // 토큰이 존재하고 유효하다고 가정
                const payload = JSON.parse(atob(token.split('.')[1]));
                this.users_name = payload.sub; // 예시로 사용자 이름 설정
                this.isAuthenticated = true;
                console.log("Authenticated user:", this.users_name); // 사용자 이름 출력
            } else {
                this.isAuthenticated = false;
                this.users_name = '';
            }
        },
        toggleSidebar() {
            this.$sidebar.displaySidebar(!this.$sidebar.showSidebar);
        },
        hideSidebar() {
            this.$sidebar.displaySidebar(false);
        },
        toggleMenu() {
            this.showMenu = !this.showMenu;
        }
    },
    created() {
        this.checkAuthentication();
    },
    watch: {
        '$route': function () {
            this.checkAuthentication();
        },
    },
    mounted() {
        this.checkAuthentication();
    }
};
</script>