<template>
  <div class="py-4 container-fluid">
    <div class=" row">
      <div class="col-12">
        <rest-table/>
      </div>
    </div>
    <div class="row">
      <div class="col-lg-8 col-12">
      </div>
    </div>
  </div>
</template>

<script>
import RestTable from './RestTable.vue';

export default {
  name: "monthly-chart",
  components: {
    RestTable,
  },
  data() {
    return {
      stats: {
        titleColor: "opacity-7 text-white",
        descColor: "text-white",
        trip: {
          title: "Today's Trip",
          desc: "145 KM",
          classIcon: "text-dark ni ni-money-coins",
        },
        health: {
          title: "Battery Health",
          desc: "99 %",
          classIcon: "text-dark ni ni-controller ",
        },
        speed: {
          title: "Average Speed",
          desc: "56 Km/h",
          classIcon: "text-dark ni ni-delivery-fast",
        },
        volume: {
          title: "Music Volume",
          desc: "15/100",
          classIcon: "text-dark ni ni-note-03",
        },
      },
    };
  },
};
</script>
