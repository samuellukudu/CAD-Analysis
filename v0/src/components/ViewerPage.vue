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
import Vue from "vue"
import mainFont from "@/assets/fonts/Roboto-LightItalic.ttf"
import aux1Font from "@/assets/fonts/NotoSansDisplay-SemiCondensedLightItalic.ttf"
import aux2Font from "@/assets/fonts/HanaMinA.ttf"
import aux3Font from "@/assets/fonts/NanumGothic-Regular.ttf"
import LayersList from "@/components/LayersList"

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
            layers.forEach(lyr => Vue.set(lyr, "isVisible", true))
            this.layers = layers
        },

        _OnCleared() {
            this.layers = null
        },

        _OnToggleLayer(layer, newState) {
            console.log("Layer toggled:", layer, "New state:", newState);
            // Log coordinates for the selected layer
            const viewer = this.$refs.viewer.GetViewer();
            const actualLayer = viewer.layers.get(layer.name);
            if (actualLayer && actualLayer.objects) {
                actualLayer.objects.forEach((obj, idx) => {
                    if (obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position) {
                        const positions = obj.geometry.attributes.position.array;
                        console.log(`Object #${idx} positions for layer ${layer.name}:`, positions);
                    }
                });
            }
            layer.isVisible = newState;
            viewer.ShowLayer(layer.name, newState);
        },

        _OnToggleAll(newState) {
            if (this.layers) {
                for (const layer of this.layers) {
                    if (layer.isVisible !== newState) {
                        this._OnToggleLayer(layer, newState)
                    }
                }
            }
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
