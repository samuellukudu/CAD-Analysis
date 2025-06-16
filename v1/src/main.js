import { createApp } from "vue"
import App from "@/App"
import "@/assets/styles/global.less"
import quasarConfig from "@/Quasar"

console.log(`DXF viewer version: ${DXF_VIEWER_VERSION}`);

const app = createApp(App)
app.use(quasarConfig)
app.mount("#app")
