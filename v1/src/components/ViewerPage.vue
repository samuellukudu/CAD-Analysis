<template>
    <q-page class="row items-stretch root">
        <div class="col relative-position">
            <slot></slot>
            <DxfViewer ref="viewer" :dxfUrl="dxfUrl" :fonts="fonts"
                       @dxf-loaded="_OnLoaded" @dxf-cleared="_OnCleared" @dxf-message="_OnMessage" />
        </div>
        <div class="col-auto layersCol">
            <LayersList :layers="layers" @toggleLayer="_OnToggleLayer" @toggleAll="_OnToggleAll"/>
        </div>
    </q-page>
</template>

<script>
import DxfViewer from "@/components/DxfViewer"
import {DxfViewer as _DxfViewer} from "dxf-viewer"
import mainFont from "@/assets/fonts/Roboto-LightItalic.ttf"
import aux1Font from "@/assets/fonts/NotoSansDisplay-SemiCondensedLightItalic.ttf"
import aux2Font from "@/assets/fonts/HanaMinA.ttf"
import aux3Font from "@/assets/fonts/NanumGothic-Regular.ttf"
import LayersList from "@/components/LayersList"
import { reactive } from 'vue'

export default {
    name: "ViewerPage",
    components: {LayersList, DxfViewer},

    props: {
        dxfUrl: {
            type: String
        }
    },

    data() {
        return {
            layers: null
        }
    },

    methods: {
        _OnLoaded() {
            const layers = this.$refs.viewer.GetViewer().GetLayers(true)
            // Initialize layers with isVisible set to false
            this.layers = layers.map(layer => reactive({ ...layer, isVisible: false }))
        },

        _OnCleared() {
            this.layers = null
        },

        _OnToggleLayer(layer, newState) {
            if (!this.$refs.viewer) return;
            const viewer = this.$refs.viewer.GetViewer();
            if (!viewer) return;

            layer.isVisible = newState;
            viewer.ShowLayer(layer.name, newState);
        },

        _OnToggleAll(newState) {
            if (!this.layers || !this.$refs.viewer) return;
            
            const viewer = this.$refs.viewer.GetViewer();
            if (!viewer) return;

            this.layers.forEach(layer => {
                layer.isVisible = newState;
                viewer.ShowLayer(layer.name, newState);
            });
        },

        _OnMessage(e) {
            let type = "info"
            switch (e.detail.level) {
            case _DxfViewer.MessageLevel.WARN:
                type = "warning"
                break
            case _DxfViewer.MessageLevel.ERROR:
                type = "negative"
                break
            }
            this.$q.notify({ type, message: e.detail.message })
        }
    },

    created() {
        this.fonts = [mainFont, aux1Font, aux2Font, aux3Font]
    }
}
</script>

<style scoped lang="less">

.root {
    .layersCol {
        border-left: #DBDBDB solid 1px;
    }
}

</style>
