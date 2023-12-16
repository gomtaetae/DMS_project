<template>
    <div class="card mb-4 col-12">
        <div class="d-flex p-0 card-body">
            <video
                class="card-video"
                :src="require('@/assets/video/cat.mp4')"
                type="video/mp4"
                controls
            ></video>
            <!-- 외부 URL -->
            <!-- <video
                    class="card-video"
                    controls
                    :src="videoUrl"></video> -->
            </div>
    </div>
</template>

<script>
export default {
    name: "VideoCard",

    data() {
        return {
            currentVideo: '../../assets/video/cat.mp4'
            // videoUrl: 'https://youtu.be/eBXN94xQmFw?si=c26LLzyjzb516FG1'
        };
    },

    methods: {
        changeVideo(newVideo) {
            this.currentVideo = newVideo;
        }
    }
};
</script>

<style>
.card-video {
    width: 100%;
    height: auto;
}
</style>