<template>
  <div class="card">
    <div class="card-header pb-0">
      <h6>휴게소 편의시설 현황</h6>
    </div>
    <div class="card-body px-0 pt-0 pb-2">
      <div class="table-responsive p-0">
        <table class="table align-items-center mb-0">
          <thead>
            <tr>
              <th class="text-uppercase text-secondary text-xs font-weight-bolder opacity-10">휴게소(주유소)명</th>
              <th
                class="text-uppercase text-secondary text-xs font-weight-bolder opacity-10"
              >주소</th>
              <th
                class="text-center text-uppercase text-secondary text-xs font-weight-bolder opacity-10"
              >쉼터 유무</th>
              <th
                class="text-center text-uppercase text-secondary text-xs font-weight-bolder opacity-10"
              >편의시설</th>
              <th
                class="text-center text-uppercase text-secondary text-xs font-weight-bolder opacity-10"
              >전화번호</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in items" :key="item.serviceAreaCode">
              <td>
                <div class="d-flex px-2 py-1">
                  <div>
                    <img
                      :src="require(`../../assets/img/shelters/shelter_${item.serviceAreaCode2}.jpg`)"
                      class="avatar avatar-sm me-3"
                      :alt="'shelter' + item.serviceAreaCode2"
                    />
                  </div>
                  <div class="d-flex flex-column justify-content-center">
                    <h6 class="mb-0 text-sm">{{ item.serviceAreaName }}</h6>
                  </div>
                </div>
              </td>
              <td>
                <p class="text-xs font-weight-bold mb-0">{{ item.svarAddr}}</p>
                <p class="text-xs text-secondary mb-0">{{ item.routeName }}</p>
              </td>
              <td class="align-middle text-center text-xl font-weight-bold">
                <div v-if="item.convenience && item.convenience.includes('쉼터')">
                  <span class="badge badge-sm bg-gradient-primary">있음</span>
                </div>
                <div v-else>
                  <span class="badge badge-sm bg-gradient-info">없음</span>
                </div>
              </td>
              <td class="align-middle text-center text-xl font-weight-bold">
                <span class="badge badge-sm bg-gradient-secondary">{{ item.brand }}</span>
              </td>
              <td class="align-middle text-center text-xl font-weight-bold">
                <div class="d-flex flex-column justify-content-center">
                  <h6 class="mb-0 text-sm">{{ item.telNo }}</h6>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: "rest-table",
  data() {
    return {
      items: []
    };
  },
  mounted() {
    axios.get('http://data.ex.co.kr/openapi/business/conveniServiceArea?key=4418561388&type=json&numOfRows=10&pageNo=1')
      .then(response => {
        this.items = response.data.list;
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }
};
</script>
