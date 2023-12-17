<template>
  <div class="card mb-4">
    <div class="pb-0 card-header mb-0">
      <h6 style="font-weight: bold; font-size: large;">운전자 졸음 시간대 예측 (월통계)</h6>
    </div>
    <div class="p-3 card-body">
      <div class="chart">
        <canvas id="day-prediction" class="chart-canvas" height="300"></canvas>
      </div>
    </div>
  </div>
</template>

<script>
import Chart from "chart.js/auto";

export default {
  name: "gradient-line-chart",

  props: {
    title: {
      type: String,
    },
    detail1: {
      type: String,
    },
    detail2: {
      type: String,
    },
  },

  mounted() {
    var ctx1 = document.getElementById("day-prediction").getContext("2d");

    var gradientStroke1 = ctx1.createLinearGradient(0, 230, 50, 0);

    gradientStroke1.addColorStop(1, "rgba(94, 114, 228, 0.2)");
    gradientStroke1.addColorStop(0.2, "rgba(94, 114, 228, 0.0)");
    gradientStroke1.addColorStop(0, "rgba(94, 114, 228, 0)");

    new Chart(ctx1, {
      type: "line",
      data: {
        labels: ["07시", "10시", "13시", "16시", "19시", "22시", "01시", "04시"],
        datasets: [
          {
            label: "졸음 예상 확률",
            tension: 0.3,
            borderWidth: 3,
            pointRadius: 0,
            borderColor: "#EB91CB",
            backgroundColor: gradientStroke1,
            // eslint-disable-next-line no-dupe-keys
            fill: true,
            data: [0, 5, 73, 44, 8, 27, 62, 14],
            maxBarThickness: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
        },
        interaction: {
          intersect: false,
          mode: "index",
        },
        scales: {
          y: {
            grid: {
              drawBorder: false,
              display: true,
              drawOnChartArea: true,
              drawTicks: false,
              borderDash: [5, 5],
            },
            ticks: {
              display: true,
              padding: 10,
              color: "#fbfbfb",
              font: {
                size: 11,
                family: "Open Sans",
                style: "normal",
                lineHeight: 2,
              },
            },
          },
          x: {
            grid: {
              drawBorder: false,
              display: false,
              drawOnChartArea: false,
              drawTicks: false,
              borderDash: [5, 5],
            },
            ticks: {
              display: true,
              color: "#ccc",
              padding: 20,
              font: {
                size: 11,
                family: "Open Sans",
                style: "normal",
                lineHeight: 2,
              },
            },
          },
        },
      },
    });
  },
};
</script>