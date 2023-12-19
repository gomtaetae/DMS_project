<template>
  <div class="py-4 container-fluid">
    <div class="row">
      <div class="col-lg-12">
        <!-- Cards -->
        <div class="row">
          <div class="col-lg-3 col-md-6 col-12">
            <card style="background-color: #F5F1D6;" :title="stats.drowsy.title" :total="stats.drowsy.total"
              :value="stats.drowsy.value" :percentage="stats.drowsy.percentage"
              :percentageColor="stats.drowsy.percentageColor" :iconBackground="stats.drowsy.iconBackground"
              :detail="stats.drowsy.detail" :iconImage="stats.drowsy.iconImage" directionReverse>
            </card>
          </div>
          <div class="col-lg-3 col-md-6 col-12">
            <card style="background-color: #F5F1D6;" :title="stats.Calling.title" :total="stats.Calling.total"
              :value="stats.Calling.value" :percentage="stats.Calling.percentage"
              :percentageColor="stats.Calling.percentageColor" :iconBackground="stats.Calling.iconBackground"
              :detail="stats.Calling.detail" :iconImage="stats.Calling.iconImage" directionReverse></card>
          </div>
          <div class="col-lg-3 col-md-6 col-12">
            <card style="background-color: #F5F1D6;" :title="stats.Texting.title" :total="stats.Texting.total"
              :value="stats.Texting.value" :percentage="stats.Texting.percentage"
              :iconBackground="stats.Texting.iconBackground" :percentageColor="stats.Texting.percentageColor"
              :detail="stats.Texting.detail" :iconImage="stats.Texting.iconImage" directionReverse></card>
          </div>
          <div class="col-lg-3 col-md-6 col-12">
            <card style="background-color: #F5F1D6;" :title="stats.Smoking.title" :total="stats.Smoking.total"
              :value="stats.Smoking.value" :percentage="stats.Smoking.percentage"
              :iconBackground="stats.Smoking.iconBackground" :detail="stats.Smoking.detail"
              :iconImage="stats.Smoking.iconImage" directionReverse></card>
          </div>
        </div>
        <!-- Video: 로그인하지 않았을 때만 보임 -->
        <div class="row" v-if="!isLoggedIn">
          <div class="col-12">
            <video-card />
          </div>
        </div>
        <!-- Charts: 로그인했을 때만 보임 -->
        <div class="row" v-if="isLoggedIn">
          <div class="col-12">
            <GradientLineChart />
          </div>
        </div>
        <div class="row" v-if="isLoggedIn">
          <div class="col-lg-8 col-12">
            <DefaultLineChart />
          </div>
          <div class="col-lg-4 col-12">
            <ThinBarChart />
          </div>
        </div>
        <div class="row mt-4">
          <div class="col-lg-7 mb-lg-0 mb-4">
            <div class="card">
              <div class="table-responsive">
                <authors-table />
              </div>
            </div>

          </div>
          <div class="col-lg-5">
            <categories-card />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
// import CategoriesCard from "./components/CategoriesCard.vue";
// import AuthorsTable from "@/views/components/AuthorsTable.vue";
import Card from "@/examples/Cards/Card.vue";
import GradientLineChart from "@/examples/Charts/GradientLineChart.vue";
import VideoCard from "@/examples/Cards/VideoCard.vue";
import DefaultLineChart from "@/examples/Charts/DefaultLineChart.vue";
import { mapState } from 'vuex';
import sleep from "@/assets/png/sleep.png";
import cellphone from "@/assets/png/cellphone.png";
import message from "@/assets/png/message.png";
import smoking from "@/assets/png/smoking.png";
import ThinBarChart from "@/examples/Charts/ThinBarChart.vue";

// axios
import axios from 'axios';

export default {
  name: "Dashboard",
  computed: {
    ...mapState(['isLoggedIn']), // Vuex 스토어에서 isLoggedIn 상태를 매핑
  },
  data() {
    return {
      my_data: '',
      ran_str: [],
      stats: {
        drowsy: {
          title: "졸음",
          total: "누적",
          value: "13",
          percentage: "1회 증가",
          percentageColor: "text-danger",
          iconBackground: "bg-gradient-warning",
          detail: "지난 달보다",
          iconImage: sleep,
        },
        Calling: {
          title: "통화",
          total: "누적",
          value: "57",
          percentage: "3회 감소",
          percentageColor: "text-info",
          iconBackground: "bg-gradient-success",
          detail: "지난 달보다",
          iconImage: cellphone,
        },
        Texting: {
          title: "문자 메시지",
          total: "누적",
          value: "123",
          percentage: "7회 증가",
          percentageColor: "text-danger",
          iconBackground: "bg-gradient-warning",
          detail: "지난 달보다",
          iconImage: message,
        },
        Smoking: {
          title: "흡연",
          total: "누적",
          value: "0",
          percentage: "0회",
          iconBackground: "bg-gray-500",
          detail: "지난 달보다",
          iconImage: smoking,
        },
      },
    };
  },
  methods: {
    getMyData() {
      let path = "http://" + window.location.hostname + ":5000/";
      axios.get(path).then((res) => {
        this.my_data = res.data;
      }).catch((error) => {
        console.error(error);
      });
    },
    getRanStr() {
      let path = "http://" + window.location.hostname + ":5000/main_btn";
      axios.get(path).then((res) => {
        this.ran_str = res.data;
        console.log(res.data);
      }).catch((error) => {
        console.error(error);
      })
    },
    checkToken() {
      const token = localStorage.getItem('token');
      if (!token) {
        console.log('No token found');
        // 로그인 페이지로 리디렉션하거나 사용자에게 알림
      }
    },
  },
  created() {
    const token = localStorage.getItem('token');
    if (token) {
      console.log('Token exists:', token);
      // 추가적인 로직 처리
    } else {
      console.log('No token found');
      // 로그인 페이지로 리디렉션하거나 사용자에게 알림
    }
    this.checkToken();
    // console.log('Dashboard created');
    // const token = localStorage.getItem('token');
    // console.log('Dashboard token:', token);

    this.getMyData();
  },
  components: {
    // AuthorsTable,
    // CategoriesCard,
    Card,
    GradientLineChart,
    DefaultLineChart,
    VideoCard,
    ThinBarChart
  },
};
</script>
