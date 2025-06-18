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
            // Initialize layers with isVisible set to true (all selected by default)
            this.layers = layers.map(layer => reactive({ ...layer, isVisible: true }))
            // Also show all layers in the viewer
            const viewer = this.$refs.viewer.GetViewer();
            if (viewer && typeof viewer.ShowLayer === 'function') {
                this.layers.forEach(layer => {
                    viewer.ShowLayer(layer.name, true);
                });
            }
        },

        _OnCleared() {
            this.layers = null
        },

        _OnToggleLayer(layer, newState) {
            if (!this.$refs.viewer) return;
            const viewer = this.$refs.viewer.GetViewer();
            if (!viewer) return;

            // Enhanced logging
            console.log("Layer toggled:", layer, "New state:", newState);
            // Diagnostic logging for entities
            if (typeof this.$refs.viewer.GetEntitiesByLayer === 'function') {
                const entities = this.$refs.viewer.GetEntitiesByLayer(layer.name);
                console.log(`[DIAG] Raw entities for layer '${layer.name}':`, entities);
                const entityTypes = entities.map(e => e.type || e.constructor?.name || 'Unknown');
                console.log(`[DIAG] Entity types for layer '${layer.name}':`, entityTypes);
            }
            // Log coordinates for the selected layer in a structured way
            let actualLayer = null;
            if (viewer.layers && typeof viewer.layers.get === 'function') {
                actualLayer = viewer.layers.get(layer.name);
            }
            let structured = [];
            if (actualLayer && actualLayer.objects) {
                actualLayer.objects.forEach((obj, objIdx) => {
                    // Handle LineSegments: split into individual lines
                    if ((obj.type || obj.constructor?.name) === 'LineSegments' && obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position) {
                        const positions = obj.geometry.attributes.position.array;
                        for (let i = 0; i < positions.length - 3; i += 4) { // Each line: [x1, y1, x2, y2]
                            structured.push({
                                index: structured.length,
                                type: 'Line',
                                coordinates: [
                                    [positions[i], positions[i + 1]],
                                    [positions[i + 2], positions[i + 3]]
                                ],
                                color: obj.color || obj.material?.color?.getHexString?.() || null,
                                layer: obj.layer || layer.name,
                                id: obj.id || obj.uuid || null,
                                handle: obj.handle || null,
                                name: obj.name || null,
                                blockReference: obj.blockReference || null,
                                parentType: 'LineSegments',
                                raw: obj
                            });
                        }
                    } else if ((obj.type || obj.constructor?.name) === 'ArcSegments' && obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position) {
                        // For ArcSegments, treat each arc as a separate component (assuming 3 points per arc: start, mid, end)
                        const positions = obj.geometry.attributes.position.array;
                        for (let i = 0; i < positions.length - 5; i += 6) { // Each arc: [x1, y1, x2, y2, x3, y3]
                            structured.push({
                                index: structured.length,
                                type: 'Arc',
                                coordinates: [
                                    [positions[i], positions[i + 1]],
                                    [positions[i + 2], positions[i + 3]],
                                    [positions[i + 4], positions[i + 5]]
                                ],
                                color: obj.color || obj.material?.color?.getHexString?.() || null,
                                layer: obj.layer || layer.name,
                                id: obj.id || obj.uuid || null,
                                handle: obj.handle || null,
                                name: obj.name || null,
                                blockReference: obj.blockReference || null,
                                parentType: 'ArcSegments',
                                raw: obj
                            });
                        }
                    } else {
                        // Fallback: treat as a single object/component
                        let coords = [];
                        if (obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position) {
                            const positions = obj.geometry.attributes.position.array;
                            for (let i = 0; i < positions.length; i += 2) {
                                coords.push([positions[i], positions[i + 1]]);
                            }
                        }
                        structured.push({
                            index: structured.length,
                            type: obj.type || obj.constructor?.name || 'Unknown',
                            coordinates: coords,
                            color: obj.color || obj.material?.color?.getHexString?.() || null,
                            layer: obj.layer || layer.name,
                            id: obj.id || obj.uuid || null,
                            handle: obj.handle || null,
                            name: obj.name || null,
                            blockReference: obj.blockReference || null,
                            parentType: obj.type || obj.constructor?.name || 'Unknown',
                            raw: obj
                        });
                    }
                });
                console.log(`Layer '${layer.name}' components:`, structured);
            } else if (typeof this.$refs.viewer.GetEntitiesByLayer === 'function') {
                // Fallback: log entities if available, structured
                const entities = this.$refs.viewer.GetEntitiesByLayer(layer.name);
                structured = entities.map((entity, idx) => ({
                    index: idx,
                    type: entity.type || entity.constructor?.name || 'Unknown',
                    coordinates: entity.vertices || entity.points || null,
                    color: entity.color || null,
                    layer: entity.layer || layer.name,
                    id: entity.id || entity.uuid || null,
                    handle: entity.handle || null,
                    name: entity.name || null,
                    blockReference: entity.blockReference || null,
                    raw: entity
                }));
                console.log(`Layer '${layer.name}' entities:`, structured);
            }

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
