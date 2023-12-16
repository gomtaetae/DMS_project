<template>
  <div class="card">
    <div class="card-header pb-0">
      <h6>Drowsy Shelter</h6>
    </div>
    <div class="card-body px-0 pt-0 pb-2">
      <div class="table-responsive p-0">
        <table class="table align-items-center mb-0">
          <thead>
            <tr>
              <th class="text-uppercase text-secondary text-xxs font-weight-bolder opacity-10">shelter name</th>
              <th
                class="text-uppercase text-secondary text-xxs font-weight-bolder opacity-10 ps-2"
              >location</th>
              <th
                class="text-center text-uppercase text-secondary text-xxs font-weight-bolder opacity-10"
              >direction</th>
              <th
                class="text-center text-uppercase text-secondary text-xxs font-weight-bolder opacity-10"
              >favorite</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in responseData.data" :key="index">
              <td>
                <div class="d-flex px-2 py-1">
                  <div>
                    <img
                      :src="require(`../../assets/img/shelters/shelter(${index + 1}).jpg`)"
                      class="avatar avatar-sm me-3"
                      :alt="'shelter' + index"
                    />
                  </div>
                  <div class="d-flex flex-column justify-content-center">
                    <h6 class="mb-0 text-sm">{{ item.명칭 }}</h6>
                  </div>
                </div>
              </td>
              <td>
                <p class="text-xs font-weight-bold mb-0">{{ item.노선}}</p>
                <p class="text-xs text-secondary mb-0">{{ item.이정 }}</p>
              </td>
              <td class="align-middle text-center text-xl font-weight-bold">
                  <span class="badge badge-sm bg-gradient-secondary">{{ item.방향 }}</span>
              </td>
              <td class="align-middle text-center">
                <img src="../../assets/svg/star-outline.svg" alt="star-outline" style="height:1.5rem;width:1.5rem" aria-hidden="true">
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
  name: "shelter-table",
  data() {
    return {
      responseData: { data: [] }
    };
  },
  mounted() {
    axios.get('https://api.odcloud.kr/api/15043710/v1/uddi:f49b8c81-af90-41f3-99b7-ed665178731a?page=1&perPage=10&returnType=JSON&serviceKey=FTASscvsf4EMTj1k1CI5zH59G1NenFJ51jTshkstqtQk5Nxpfgd0uPmEy2Wv43H%2FkMMK3GfPXuq24oDc8t3MGw%3D%3D')
      .then(response => {
        this.responseData.data = response.data.data.map(item => item);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }
};
</script>