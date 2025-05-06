# AgricultureKnowledgeBase

## High level architecture

```sql
+--------------------+
|   User Interface   |
| (chatbot/search UI)|
+--------------------+
         ↓
+------------------------+
|   Query Preprocessing  |
| (intent parsing, NER)  |
+------------------------+
         ↓
+-----------------------------+
| Retrieval Augmented Gen.   |
|  (LLM + Vector Store)       |
+-----------------------------+
     ↓           ↑
+---------+   +----------------+
|  LLM    |←→| Vector DB (FAISS/|
| (e.g.   |   | Weaviate, etc.)|
| GPT-4)  |   +----------------+
+---------+         ↑
         ↓      +-------------------------+
   +-------------| Knowledge Base         |
   |             | (extracted facts, docs)|
   |             +-------------------------+
   ↓
+----------------------+
| Response Generation  |
| (Natural language)   |
+----------------------+
```

Knowledge base for agriculture, supporting both images and texts

* **Intelligent Information Retrieval** to help users find answers quickly?
* **Automated Knowledge Extraction** to build and maintain the knowledge base efficiently?

* **Visual Search Enhancement:** Could a VLM help users find information by describing or showing an image? For example, a farmer could upload a picture of a plant and ask, "What disease is this?"
* **Image Understanding in Knowledge Articles:** If the knowledge base contains articles or guides with images, could a VLM automatically generate descriptions or extract key information from those visuals?

## Datasets

Here are some searches I'll perform to find datasets for:
1.  **Intelligent Information Retrieval in Agriculture:** Datasets that could be used to train models to understand and respond to agricultural queries.
2.  **Automated Knowledge Extraction in Agriculture:** Datasets containing agricultural texts or structured data from which knowledge graphs or databases could be built.
3.  **Predictive Modeling in Agriculture:** Datasets with historical agricultural data (e.g., crop yields, weather, soil data) for forecasting.
4.  **Pest and Disease Identification in Agriculture:** Image datasets of plant pests and diseases, ideally with labels. This is a prime area for VLMs.
5.  **Visual Search Enhancement & Image Understanding in Agricultural Knowledge Articles:** Datasets that combine images with textual descriptions or questions in the agricultural context.

Let's start gathering some potential data sources.

Based on the search results, here's a breakdown of datasets related to your specified tasks in agriculture, including those suitable for VLMs:

**1. Intelligent Information Retrieval & Automated Knowledge Extraction:**
*   **Agricultural Question-Answering Datasets:**
    *   `KisanVaani/agriculture-qa-english-only` (Hugging Face): Contains English question-answer pairs covering crop production, animal husbandry, soil management, and farming practices, curated from online agricultural forums and websites. Useful for training Q&A and information retrieval models.
    *   `Agronomic Question and Answer Dataset` (Kaggle): Focuses on beans, maize, and cassava, covering topics like cultivation, pest management, soil health, and harvesting. Serves as a knowledge base.
    *   `MilkQA`: Contains 2,657 question-answer pairs in Portuguese related to agriculture, collected by the Brazilian Agricultural Research Corporation (Embrapa).
*   **Agricultural Text Datasets:**
    *   Sources like agricultural research papers, specialist websites (e.g., Encyclopedia), and agricultural books contain unstructured or semi-structured data. These require preprocessing (cleaning, annotation) for knowledge extraction.
    *   Models like AES-BERCNN have been developed using self-constructed agricultural datasets for text classification in expert systems.
*   **Knowledge Graph Focused Datasets/Projects:**
    *   `AgriNER`: An NER dataset with 36 agricultural entity types and 9 relation types extracted from research papers, designed for building agricultural knowledge graphs.
    *   Potato Diseases and Pests Knowledge Graph: A project constructing a knowledge base using data crawled from agricultural websites and books, focusing on entity and relation extraction.
    *   Agricultural Knowledge Graph Dataset (China): Includes data on 11 agricultural categories (grains, fruits, vegetables, pests/diseases, etc.) totaling 8,481 subcategories, used to build a knowledge graph with 90,508 triplets.
    *   Embedding-based retrieval with LLM work explored extracting structured data (entities, attributes) from unlabeled agricultural documents using LLMs and embedding retrieval.
    *   D2KAB Project: Focuses on extracting and formalizing knowledge from agronomy and biodiversity data using Semantic Web technologies to create FAIR knowledge graphs.
*   **General Agricultural Databases:**
    *   `FAOSTAT`: A comprehensive global statistical database from the UN's Food and Agriculture Organization covering production, trade, land use, etc..
    *   `USDA NASS`: Detailed data on US agriculture (yields, practices, economics).
    *   `Ecoinvent Database (Agriculture Sector)`: Contains datasets on crop production (perennial, non-perennial, organic, greenhouse), animal production, agricultural services (tillage, harvesting), and processing.

**2. Predictive Modeling in Agriculture:**
*   **Crop Yield Prediction:**
    *   `CropNet Dataset`: A large-scale, multi-modal dataset (Sentinel-2 imagery, weather data, USDA crop data) for climate-change-aware crop yield prediction in the US.
    *   `Crop Yield Prediction Dataset` (Kaggle): Contains historical crop yields, weather conditions, and soil characteristics.
    *   `Agricultural Crop Yield in Indian States Dataset` (Kaggle): Data from 1997-2020 for multiple Indian crops, including yield, season, state, area, production, rainfall, fertilizer, and pesticide usage.
    *   Remote Sensing for Yield Prediction (GitHub Project): Uses FAO, World Bank, and Earth Engine data (yield, temperature, pesticides, land area, fertilizer, rainfall, MODIS imagery) to predict yields, specifically soybeans in Illinois using CNN-LSTM.
    *   USDA Datasets (2003-2013): Used in research to forecast crop yields using models like Random Forest and XGBoost, analyzing factors affecting output.
    *   `Synthetic Agricultural Yield Prediction Dataset` (Kaggle): Simulated data with soil quality, seed variety, fertilizer, sunny days, rainfall, and yield, including outliers for model robustness testing.
    *   `Smart Farming Sensor Data for Yield Prediction` (Kaggle).
    *   `Louisiana Sugarcane yield data (2010-2021)` (Kaggle).
*   **Crop Recommendation & Suitability:**
    *   `Crop Recommendation Dataset` (Kaggle): Uses soil N, P, K ratios, temperature, humidity, pH, and rainfall to recommend suitable crops in India.
    *   `Crop Recommendation using Soil Properties and Weather Prediction Dataset` (Mendeley Data): Combines soil data (pH, color, composition, nutrients) from Ethiopia's ATA and climate data (temp, precipitation, humidity, etc.) from NASA for crop recommendation.
    *   `Crop and Soil DataSet` (Kaggle): Designed for crop recommendation based on soil properties (N, P, K, pH) and potentially regional climate data.
    *   `Project: Predictive Modeling for Agriculture` (GitHub): Uses soil N, P, K, and pH (`soil_measures.csv`) to predict the best crop using multi-class classification.
    *   `CropSuite Model Datasets`: Uses climate data (CHIRPS, CHIRTS), soil data (SoilGrids), and terrain data to simulate crop suitability for 48 crops in Africa, considering climate variability.
    *   European Soil Data Centre (ESDAC): Offers land suitability maps for various crops in Europe based on eco-pedological indicators.
*   **Other Predictive Tasks:**
    *   General predictive analytics rely on data from sensors, equipment, weather records, and imagery (satellite/drone). Models like ARIMA, Prophet, Random Forests, and RNNs are used.
    *   Soil Moisture Active Passive (SMAP) Dataset: NASA satellite data for global soil moisture, useful for irrigation management and drought forecasting.

**3. Pest and Disease Identification (Excellent for VLMs):**
*   **Image Datasets:**
    *   `PlantVillage Dataset`: One of the largest leaf image datasets (healthy/diseased), covering 38 disease classes across 14 crops. Contains over 54,000 images.
    *   `PlantDoc Dataset`: High-quality images of 27 plant disease classes, often used in research but potentially limited by controlled environment collection.
    *   `Dataset for Crop Pest and Disease Detection` (Mendeley Data / Kaggle): Contains ~25k raw images and ~103k augmented images from Ghana covering Cashew, Cassava, Maize, and Tomato (22 classes total), validated by experts.
    *   `Pest and Disease Photo Database` (University of Delaware): Features photos of pests/diseases impacting vegetable and fruit crops in Delaware.
    *   `Taiwan Tomato Leaves Dataset` (Kaggle): Suitable for tomato plant disease classification and prediction.
    *   `Agarwood Pest and Disease Dataset (APDD)` & `Turkey Plant Pests and Diseases (TPPD)`: Used in research presenting a lightweight DL model, APDD has 5,472 images (14 classes), TPPD has 4,447 images (15 classes across 6 plants).
    *   `Guava Leaf Disease Dataset`: 4046 images across 7 disease classes used for comparing traditional ML and DL methods.
    *   `Rice Leaf Diseases Augmented New Dataset` (Kaggle).
    *   `Apple Leaf Disease Classification Dataset` (Kaggle).
    *   Various other plant disease datasets mentioned across sources.
*   **Aerial/Drone Imagery:**
    *   `Agriculture-Vision Dataset`: Over 94,000 annotated aerial images of fields showing anomalies like weeds, dry areas, and insect damage. An extended version includes full-field imagery.
*   **Datasets used in VLM/LMM Research:**
    *   `AgroInstruct`: A pipeline curated expert-level instruction-tuning data from six image-only datasets (plant diseases, weeds, farm insects, fruits) to enhance LMMs for agriculture.
    *   `Agri-LLaVA Datasets`: Includes 391,785 image-text pairs on pests/diseases and a curated multimodal knowledge-based instruction-tuning dataset for fine-tuning agricultural LMMs.

**4. Visual Search Enhancement & Image Understanding (VLM-focused):**
*   **Visual Question Answering (VQA) & Multimodal Datasets:**
    *   `AgMMU`: A comprehensive benchmark with 5,460 multiple-choice & open-ended questions derived from user-expert conversations, plus `AgBase-200K`, a knowledge base of 205,399 facts for fine-tuning VLMs on agricultural tasks (pest ID, disease categorization, etc.).
    *   `AgroEvals`: A VQA framework built using test sets from datasets used for AgroInstruct, testing model capabilities at different levels (e.g., presence of disease, identifying general categories).
    *   `WheatRustDL2024` & `WheatRustVQA`: Custom datasets created for VQA on wheat rust detection. `WheatRustDL2024` has ~8k images, and `WheatRustVQA` has 1800 augmented images with Q&A pairs.
    *   `Agri-LLaVA VQA Dataset`: A manually annotated VQA dataset with 482 images (pests, diseases, healthy) and 2,268 Q&A pairs (4-5 rounds per image) created specifically for testing their agricultural LMM.
    *   `FieldSAFE`: A multi-modal dataset for obstacle detection in agriculture.
    *   Research on Multimodal Knowledge Extraction and Q&A in Farming mentions creating synthetic VQA datasets due to a lack of existing ones and extending agricultural knowledge graphs (like AGROVOC).
    *   `AgriVLM` research used datasets for crop disease recognition and growth stage recognition, achieving high accuracy by fusing image and text data.
*   **General Image Datasets with Agricultural Relevance:**
    *   `Open Images Dataset for Agriculture`: A subset of Open Images annotated for agricultural objects (machinery, animals, crops).
    *   `Sentinel-2 Satellite Imagery`: High-resolution multispectral imagery useful for crop monitoring, health analysis, and mapping.
    *   GitHub Repositories (e.g., `ricber/digital-agriculture-datasets`): Curated lists often include datasets spanning RGB, multispectral, hyperspectral, and LiDAR data for tasks like classification, segmentation, and navigation.
    *   Roboflow Universe: Hosts thousands of datasets, including specific ones for weeds, apple sorting, livestock detection, etc..

**Where to Find Datasets:**
*   **Platforms:** Kaggle, Hugging Face Datasets, Radiant MLHub, Papers With Code, Mendeley Data, Roboflow Universe, GitHub.
*   **Organizations:** FAO (FAOSTAT, FAM Catalogue), USDA NASS, CGIAR Big Data Platform, ESA (Sentinel data), NASA (SMAP, climate data), European Soil Data Centre (ESDAC), Lacuna Fund, National Agricultural Science Data Centre (China), Ethiopian Agricultural Transformation Agency (ATA).
*   **Research Projects/Papers:** Often release datasets (e.g., PlantVillage, Agriculture-Vision, AgMMU, Agri-LLaVA, CropNet, AgriNER).
*   
## Agriculture datasets in China
以下是关于中国农业相关的**权威网站、核心数据集**和**官方政府报告**的整理，涵盖农业政策、生产统计、市场监测、土地资源、科技创新等领域，适合学术研究、政策分析或商业决策参考：

---

### **一、核心政府部门与官方网站**
1. **农业农村部**  
   - 网站: [www.moa.gov.cn](http://www.moa.gov.cn)  
   - 职能：主管全国农业农村经济发展，发布农业政策、统计数据和行业报告。  
   - 重点栏目：  
     - **数据开放**：国家农业普查、农产品价格监测、农村经济统计。  
     - **政策法规**：中央一号文件、乡村振兴战略规划等。  
     - **市场信息**：农产品供需平衡表、农业市场预警系统。  

2. **国家统计局**  
   - 网站: [www.stats.gov.cn](http://www.stats.gov.cn)  
   - 职能：发布全国农业宏观经济数据。  
   - 核心数据：  
     - **《中国统计年鉴》**：农业产值、播种面积、粮食产量等宏观指标。  
     - **年度农业统计公报**：粮食总产量、农民收入等关键数据。  

3. **自然资源部**  
   - 网站: [www.mnr.gov.cn](http://www.mnr.gov.cn)  
   - 职能：管理耕地资源、土地利用规划。  
   - 数据资源：  
     - **全国耕地质量等级报告**  
     - **土地利用现状遥感监测数据**  

4. **生态环境部**  
   - 网站: [www.mee.gov.cn](http://www.mee.gov.cn)  
   - 职能：农业面源污染治理、生态保护。  
   - 报告：《中国生态环境状况公报》（含农业污染数据）。  

---

### **二、核心农业数据集**
1. **国家农业统计数据库**  
   - **数据来源**：农业农村部、国家统计局  
   - **覆盖内容**：  
     - 农作物产量（粮食、蔬菜、水果）、畜牧业数据（生猪、禽类存栏量）、渔业产量。  
     - 农民人均收入、农村固定资产投资。  
   - **获取方式**：  
     - 农业农村部官网“数据开放”栏目  
     - 国家统计局“国家数据库”（[data.stats.gov.cn](http://data.stats.gov.cn)）  

2. **农业资源环境监测数据**  
   - **耕地质量监测**：农业农村部“耕地质量保护与提升”项目，覆盖土壤肥力、酸化盐渍化数据。  
   - **农业气象数据**：中国气象局与农业农村部联合发布的农业气象灾害预警。  

3. **农产品市场数据**  
   - **农业农村部市场预警系统**：  
     - 每日更新“全国农产品批发市场价格信息系统”（[nytb.moa.gov.cn](http://nytb.moa.gov.cn)）。  
     - 月度《中国农产品供需形势分析》（CASDE）。  
   - **海关总署**：农产品进出口数据（[www.customs.gov.cn](http://www.customs.gov.cn)）。  

4. **农村改革与土地数据**  
   - **农村土地承包经营权确权登记数据**：农业农村部牵头的全国农村土地确权数据库。  
   - **农村集体资产清产核资数据**：2021年全国农村集体资产清查结果（部分公开）。  

---

### **三、权威官方报告**
1. **中央一号文件（年度农业农村政策纲领）**  
   - 发布时间：每年初  
   - 内容重点：粮食安全、乡村振兴、农业科技、农民增收等政策方向。  
   - 获取路径：农业农村部官网或中国政府网（[www.gov.cn](http://www.gov.cn)）。  

2. **《中国农业发展报告》（农业农村部发布）**  
   - 年度报告：涵盖农业经济形势、政策实施效果、未来趋势分析。  
   - 获取方式：农业农村部官网“政策法规”栏目。  

3. **《中国粮食安全白皮书》**  
   - 发布机构：国务院新闻办公室  
   - 内容：中国粮食安全政策、生产能力、储备体系等（最新版为2023年）。  

4. **《中国乡村振兴战略规划（2018-2022年）》及年度进展报告**  
   - 发布机构：国家发改委、农业农村部  
   - 内容：农村产业、生态、文化、治理等领域的阶段性成果。  

5. **《中国农业绿色发展报告》**  
   - 发布机构：农业农村部农业生态与资源保护总站  
   - 内容：农业资源利用效率、面源污染治理、可持续发展评估。  

---

### **四、地方农业数据与报告**
1. **省级农业农村厅网站**  
   - 示例：  
     - **北京市农业农村局**：[nyncj.beijing.gov.cn](http://nyncj.beijing.gov.cn)  
     - **黑龙江省农业农村厅**：[nynct.heilongjiang.gov.cn](http://nynct.heilongjiang.gov.cn)  
   - 特点：提供地方特色农业数据（如东北玉米、南方水稻产量）、区域政策文件。  

2. **地方统计年鉴**  
   - 各省、市、县统计局发布《统计年鉴》，包含本地农业经济详细数据。  

3. **农村固定观察点数据**  
   - 农业农村部在全国设立的200个农村固定观察点，长期跟踪农村经济和社会变化（数据需申请使用）。  

---

### **五、特色资源与工具**
1. **农业遥感监测**  
   - **农业农村部遥感应用中心**：利用卫星监测农作物长势、病虫害和灾情（数据部分公开）。  
   - **国家对地观测科学数据中心**：[www.dsac.cn](http://www.dsac.cn)（提供农业土地利用遥感数据）。  

2. **农业基因组与种业数据**  
   - **中国农业科学院**：[www.caas.cn](http://www.caas.cn)  
   - 数据集：作物基因组数据库（如水稻、小麦）、国家种质资源库。  

3. **农产品价格与期货市场数据**  
   - **大连商品交易所（DCE）**：提供玉米、大豆等农产品期货价格数据（[www.dce.com.cn](http://www.dce.com.cn)）。  

---

### **六、数据获取与使用建议**
1. **验证数据来源**：优先选择农业农村部、国家统计局等官方渠道，避免引用第三方非权威机构数据。  
2. **交叉验证**：结合多部门数据（如统计局产量数据与海关进出口数据）进行综合分析。  
3. **政策解读**：关注农业农村部官网发布的“政策解读”栏目，了解政策背景与实施细则。  
4. **学术支持**：  
   - 中国知网（CNKI）：搜索《中国农村经济》《农业经济问题》等核心期刊。  
   - 国家农业图书馆：[www.calis.net.cn](http://www.calis.net.cn)（免费获取农业文献）。  

如需特定领域（如有机农业、智慧农业、粮食安全）的深入数据或报告，请进一步说明！
