<template>
    <div class="card mb-4">
        <div class="pb-0 card-header mb-0">
            <h6 style="font-weight: bold; font-size: large;">월별 위험 행동 횟수 그래프</h6>
        </div>
        <div class="p-3 card-body">
            <div class="chart">
                <canvas id="monthly-line" class="chart-canvas" height="300"></canvas>
            </div>
        </div>
    </div>
</template>

<script>
import Chart from "chart.js/auto";

export default {
    name: "monthly-line-chart",
    mounted() {
        var ctx = document.getElementById("monthly-line").getContext("2d");
        new Chart(ctx, {
            type: "line",
            data: {
                labels: ['1월', '2월', '3월', '4월', '5월', '6월',
        '7월', '8월', '9월', '10월', '11월', '12월'],
                datasets: [
                    {
                        label: '졸음',
                        data: [0, 0, 0, 1, 3, 2,
                    0, 0, 3, 0, 3, 1],
                        tension: 0,
                        borderWidth: 3,
                        pointRadius: 3,
                        borderColor: "blue"
                    },
                    {
                        label: '통화',
                        data: [6, 3, 5, 4, 4, 6,
                    4, 2, 15, 4, 18, 15],
                        tension: 0,
                        borderWidth: 3,
                        pointRadius: 3,
                        borderColor: "green",
                    },
                    {
                        label: '핸드폰 조작',
                        data: [8, 3, 11, 9, 7, 14,
                    3, 17, 13, 6, 12, 19],
                        tension: 0,
                        borderWidth: 3,
                        pointRadius: 3,
                        borderColor: "red",
                    },
                    {
                        label: '흡연',
                        data: [0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0],
                        tension: 0,
                        borderWidth: 3,
                        pointRadius: 3,
                        borderColor: "purple",
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
                            color: "#343a40",
                            font: {
                                size: 13,
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
                            color: "#343a40",
                            padding: 20,
                            font: {
                                size: 13,
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