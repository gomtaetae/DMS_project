<template>
  <div class="card">
    <div class="p-3 pb-0 card-header">
      <h6 class="mb-0">{{ cardTitle }}</h6>
    </div>
    <div class="p-3 card-body">
      <ul class="list-group" :class="this.$store.state.isRTL ? 'pe-0' : ''">
        <li
          class="mb-2 border-0 list-group-item d-flex justify-content-between border-radius-lg"
          :class="this.$store.state.isRTL ? 'pe-0' : 'ps-0'"
        >
          <div class="d-flex align-items-center">
            <div class="text-center shadow icon icon-shape icon-sm bg-gradient-dark me-3">
              <i class="mdi mdi-sleep text-center text-lg text-white"></i>
            </div>
            <div class="d-flex flex-column">
              <h6 class="mb-1 text-sm text-dark">{{ title }}</h6>
              <span class="text-xs">
                {{ titleDesc }},
                <span class="font-weight-bold">{{ titleDesc2 }}</span>
              </span>
            </div>
          </div>
          <div class="d-flex">
            <button
              class="my-auto btn btn-link btn-icon-only btn-rounded btn-sm text-dark icon-move-right"
            >
              <i class="mdi mdi-panorama text-black text-lg" aria-hidden="true"></i>
            </button>
          </div>
        </li>
        <li
          class="mb-2 border-0 list-group-item d-flex justify-content-between border-radius-lg"
          :class="this.$store.state.isRTL ? 'pe-0' : 'ps-0'"
        >
          <div class="d-flex align-items-center">
            <div
              class="text-center shadow icon icon-shape icon-sm bg-gradient-dark me-3">
              <i class="mdi mdi-cellphone text-lg text-white text-center"></i>
            </div>
            <div class="d-flex flex-column">
              <h6 class="mb-1 text-sm text-dark">{{ title2 }}</h6>
              <span class="text-xs">
                {{ title2Desc }},
                <span class="font-weight-bold">{{ title2Desc2 }}</span>
              </span>
            </div>
          </div>
          <div class="d-flex">
            <button
              class="my-auto btn btn-link btn-icon-only btn-rounded btn-sm text-dark icon-move-right"
            >
              <i class="mdi mdi-panorama text-lg" aria-hidden="true"></i>
            </button>
          </div>
        </li>
        <li
          class="border-0 list-group-item d-flex justify-content-between border-radius-lg"
          :class="this.$store.state.isRTL ? 'pe-0' : 'ps-0'"
        >
          <div class="d-flex align-items-center">
            <div
              class="text-center shadow icon icon-shape icon-sm bg-gradient-dark me-3">
              <i class="mdi mdi-message text-center text-lg text-white"></i>
            </div>
            <div class="d-flex flex-column">
              <h6 class="mb-1 text-sm text-dark">{{ title3 }}</h6>
              <span class="text-xs">
                {{ title3Desc }},
                <span class="font-weight-bold text-danger opacity-7">{{ title3Desc2 }}</span>
              </span>
            </div>
          </div>
          <div class="d-flex">
            <button
              class="my-auto btn btn-link btn-icon-only btn-rounded btn-sm text-dark icon-move-right"
            >
              <i class="mdi mdi-panorama text-lg" aria-hidden="true"></i>
            </button>
          </div>
        </li>
        <li
          class="border-0 list-group-item d-flex justify-content-between border-radius-lg"
          :class="this.$store.state.isRTL ? 'pe-0' : 'ps-0'"
        >
          <div class="d-flex align-items-center">
            <div
              class="text-center shadow icon icon-shape icon-sm bg-gradient-dark me-3">
              <i class="mdi mdi-smoking text-center text-lg text-white"></i>
            </div>
            <div class="d-flex flex-column">
              <h6 class="mb-1 text-sm text-dark">{{ title4 }}</h6>
              <span class="text-xs font-weight-bold">{{ title4Desc }}</span>
            </div>
          </div>
          <div class="d-flex">
            <button
              class="my-auto btn btn-link btn-icon-only btn-rounded btn-sm text-dark icon-move-right"
            >
              <i class="mdi mdi-panorama text-lg" aria-hidden="true"></i>
            </button>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: "categories-card",
  props: {
    cardTitle: {
      type: String,
      default: "Bad Habit Gallery",
    },
    title: {
      type: String,
      default: "Drowsy",
    },
    title2: {
      type: String,
      default: "Calling",
    },
    title3: {
      type: String,
      default: "Texting",
    },
    title4: {
      type: String,
      default: "Smoking",
    },
    titleDesc: {
      type: String,
      default: "1 is update",
    },
    titleDesc2: {
      type: String,
      default: "TOTAL 13 Pictures",
    },
    title2Desc: {
      type: String,
      default: "3 is update",
    },
    title2Desc2: {
      type: String,
      default: "TOTAL 57 Pictures",
    },
    title3Desc: {
      type: String,
      default: "7 is update",
    },
    title3Desc2: {
      type: String,
      default: "TOTAL 123 Pictures",
    },
    title4Desc: {
      type: String,
      default: "no update",
    },
    title4Desc2: {
      default: "TOTAL 0 Pictures",
    }
  },
};
</script>