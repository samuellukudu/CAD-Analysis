<template>

<q-scroll-area class="root">
    <q-list dense>
        <q-item-label header>Layers</q-item-label>
        <q-item v-if="layers !== null" tag="label">
            <q-item-section side top>
                <q-checkbox v-model="showAll" @update:model-value="_ToggleAll"/>
            </q-item-section>
            <q-item-section>
                <q-item-label class="text-italic">All layers</q-item-label>
            </q-item-section>
        </q-item>
        <q-item v-if="layers !== null" v-for="layer in layers" :key="layer.name" tag="label">
            <q-item-section side class="q-pa-none">
                <q-icon name="label" :style="{color: _GetCssColor(layer.color)}" />
            </q-item-section>
            <q-item-section side top>
                <q-checkbox v-model="layer.isVisible" @update:model-value="(e) => _ToggleLayer(layer, e)"/>
            </q-item-section>
            <q-item-section>
                <q-item-label>{{layer.displayName}}</q-item-label>
            </q-item-section>
            <q-item-section side top>
                <q-btn size="sm" icon="download" flat @click.stop="_ExportLayerData(layer)" title="Export layer data" />
            </q-item-section>
        </q-item>
    </q-list>
</q-scroll-area>

</template>

<script>

export default {
    name: "LayersList",

    props: {
        layers: {
            /* Expecting array of {name: string, color: number, isVisible: boolean} */
            type: Array,
            default: null
        }
    },

    watch: {
        layers() {
            this.showAll = null
        }
    },

    data() {
        return {
            showAll: null
        }
    },

    methods: {
        _ToggleLayer(layer, newState) {
            console.log('Layer toggled:', {
                name: layer.name,
                displayName: layer.displayName,
                color: layer.color,
                isVisible: newState,
                data: layer
            });
            // Log geometric entities for this layer
            try {
                // Traverse up to the DxfViewer component via $parent chain
                let parent = this.$parent;
                while (parent && !parent.$refs?.viewer) {
                    parent = parent.$parent;
                }
                if (parent && parent.$refs && parent.$refs.viewer) {
                    const entities = parent.$refs.viewer.GetEntitiesByLayer(layer.name);
                    console.log('Geometric entities for layer', layer.name, entities);
                }
            } catch (e) {
                console.warn('Could not fetch entities for layer', layer.name, e);
            }
            this.$emit("toggleLayer", layer, newState)
            // Only update showAll if all layers match the new state
            this.showAll = this.layers.every(l => l.isVisible === newState)
        },

        _ToggleAll(newState) {
            console.log('All layers toggled:', {
                state: newState,
                layers: this.layers
            });
            this.showAll = newState
            this.$emit("toggleAll", newState)
        },

        _GetCssColor(value) {
            let s = value.toString(16)
            while (s.length < 6) {
                s = "0" + s
            }
            return "#" + s
        },

        _ExportLayerData(layer) {
            try {
                // Traverse up to the DxfViewer component via $parent chain
                let parent = this.$parent;
                while (parent && !parent.$refs?.viewer) {
                    parent = parent.$parent;
                }
                if (parent && parent.$refs && parent.$refs.viewer) {
                    const data = parent.$refs.viewer.GetEntitiesCoordinatesByLayer(layer.name);
                    console.log('Exported coordinates for layer', layer.name, data);
                }
            } catch (e) {
                console.warn('Could not export entities for layer', layer.name, e);
            }
        }
    }
}

</script>

<style scoped lang="less">

.root {
    height: 100%;
    max-height: 100%;
    width: 300px;
}

</style>
