# 🌟 James Li — Data Analytics & AI Portfolio

Welcome to my project portfolio!  
I specialize in data analytics, statistical modeling, and AI-driven insights using **R**, **Python**, **SQL**, **AI**, and **Dashboard**.  
Click each section below to explore projects by language or domain.

---

## 🧮 R Projects

<details>
<summary>📘 Click to view R-based projects</summary>

---

### [R-Project 1: Superstore Sales, Profit & Forecasting Analysis (R)](https://github.com/FANXYBIN/Project1_Superstore_Dataset)

<details>
<summary>📘 Click to view R-based projects</summary>

This project analyzes the **Superstore dataset (2011–2014)** to uncover insights about sales, profit, product performance, customer segments, and regional trends using **R**.  
It includes extensive **data cleaning**, **EDA visualizations**, **correlation analysis**, **boxplots**, and **ARIMA forecasting** for predicting 2015 sales and profit.

* **Dataset:** Superstore Dataset (Kaggle, 2011–2014)  
* **Tools:** R, tidyverse, ggplot2, corrplot, treemap, forecast  
* **Techniques:** Data wrangling, bar charts, scatterplots, density plots, tree maps, correlation matrices, boxplots, time-series forecasting  
* **Goal:** Understand the superstore’s historical performance and forecast next-year trends.

---

### 📁 Dataset Overview

The dataset contains 24+ columns including:

- **Sales, Profit, Discount, Quantity, Shipping Cost**  
- **Category, Sub-Category, Shipping Mode**  
- **Market, Region, Segment**  
- **Order & Ship Dates**

<div align="center">
  <img src="images/superstore_datatypes_details.png" width="650"/>
  <p><em>Datatypes descriptions of the dataset.</em></p>
</div>

---

### 🧹 Data Cleaning & Preparation

#### ✔ Checked data types and structure  
#### ✔ Identified missing values  
Only “Postal Code” contains NA values — removed for analysis.

#### ✔ Created a clean dataset `new_superstore`  
Contains only rows/columns with **no missing values**.

<div align="center">
  <img src="images/superstore_na_check.png" width="600"/>
  <p><em>NA inspection and clean dataset creation.</em></p>
</div>

---

### 📊 Orders & Customer Behavior

#### Total Orders by Market, Region, Ship Mode & Segment

<div align="center">
  <img src="images/superstore_orders.png" width="600"/>
  <p><em>Total Orders by Market, Region, Ship Mode & Segment</em></p>
</div>

- **APAC** and **Central** region show the highest order volumes.

---

### 💰 Sales & Profit by Market & Region

<div align="center">
  <img src="images/superstore_sales.png" width="600"/>
</div>


- **APAC** market and **Central** region perform best  
- **Canada** shows significantly lower sales & profit.

---

### 📦 Product Category & Subcategory Analysis

#### Orders by Category / Subcategory

<div align="center">
  <img src="images/superstore_category_counts.png" width="600"/>
</div>

Top subcategories with highest order count:

- **Binders** (12%)  
- **Storage**  
- Lowest: **Tables** (~2%)

#### Pie Chart of Subcategory Orders

<div align="center">
  <img src="images/superstore_pie_subcat.png" width="550"/>
</div>

---

### 🧭 Sales & Profit by Category / Subcategory

<div align="center">
  <img src="images/superstore_sales&profits_subcategory.png" width="600"/>
</div>


**Top findings:**

- **Phones** → highest sales  
- **Copiers** → highest profit  
- **Tables** → incurred losses  

---

### 🌳 Treemap Visualization

<div align="center">
  <img src="images/superstore_treemap.png" width="650"/>
  <p><em>Sales treemap by Category → Sub-Category</em></p>
</div>

---

### 🔍 Scatterplots

<div align="center">
  <img src="images/superstore_scatter.png" width="600"/>
</div>

Observations:

- Sales vs Shipping Cost → **positive relationship**  
- Sales vs Profit → **weak linear relationship**  
- Discount heavily reduces profit  

---

### 📈 Density Plots

<div align="center">
  <img src="images/superstore_density.png" width="600"/>
</div>

- Sales & Discount → right-skewed  
- Profit → slightly left-skewed  
- Quantity → peak around 1–2 units  

---

### 🔗 Correlation Analysis

<div align="center">
  <img src="images/superstore_corrplot.png" width="650"/>
</div>

Key relationships:

- Sales ↔ Shipping Cost → **strong positive correlation**  
- Profit ↔ Discount → **negative correlation**  

---

### 📦 Boxplots

<div align="center">
  <img src="images/superstore_box_sales.png" width="600"/>
</div>

<div align="center">
  <img src="images/superstore_box_quantity.png" width="600"/>
</div>

- Many outliers in Sales / Quantity / Discount  
- Quantity ranges from **1 → 14**  

---

### 📅 Monthly & Annual Trends

#### Annual Sales / Profit (2011–2014)

<div align="center">
  <img src="images/superstore_annual_sales&profits.png" width="600"/>
</div>

Both Sales & Profit show **yearly upward trends**.

---

#### 📆 Monthly Sales / Profit

<div align="center">
  <img src="images/superstore_monthly_sales.png" width="600"/>
</div>

<div align="center">
  <img src="images/superstore_monthly_profit.png" width="600"/>
</div>

---

### 🔮 ARIMA Forecasting (2015)

Using monthly aggregated data:

- Fitted **Auto ARIMA** for Sales & Profit  
- Forecasted next 12 months  
- Confidence interval shown in grey

<div align="center">
  <img src="images/superstore_arima_sales.png" width="650"/>
</div>

<div align="center">
  <img src="images/superstore_arima_profit.png" width="650"/>
</div>

**Forecast Insight:**  
➡ 2015 sales and profit will continue rising, with seasonality similar to previous years.

---

### 🧠 Key Insights

- **APAC + Central** regions drive the most revenue  
- **Phones** (Technology) generate the highest sales  
- **Copiers** generate the highest profit  
- **Tables** consistently lose money  
- Discount negatively impacts profit  
- Yearly performance improves steadily  
- ARIMA forecast predicts **continued growth** in 2015  

---

### 🧠 Skills Demonstrated

- R data cleaning & wrangling  
- Visual analytics (bar charts, scatterplots, boxplots, density plots)  
- Correlation analysis & treemaps  
- Time series modeling with ARIMA  
- Interpreting patterns & forecasting future performance  

</details>

</details>

---

### [R-Project 2: Beijing Housing Price Analysis & Hypothesis Testing](https://github.com/FANXYBIN/R-Project-2_Beijing_Housing_Price_Dataset)

<details>
<summary>📊 Click to view project details</summary>

This project analyzes **housing prices in Beijing** using R.  
The workflow includes extensive **data cleaning**, **EDA visualizations**, **correlation analysis**, and **hypothesis testing** to understand the key factors influencing house prices.

* **Dataset:** Beijing Housing Price (Kaggle – 318,851 observations, 26 variables)  
* **Tools:** R, tidyverse, ggplot2, corrplot, dplyr  
* **Techniques:** Data cleaning, variable recoding, histograms, correlation analysis, boxplots, scatterplots, t-tests  
* **Goal:** Explore the structure of Beijing’s housing market and validate insights using statistical hypothesis testing.

---

### 📁 Dataset Overview

The dataset contains housing transaction records from Beijing, including:

- Pricing (totalPrice, price per m²)  
- House characteristics (square, rooms, floor, buildingType, structure)  
- Location information (Lng/Lat, district, communityAverage)  
- Accessibility indicators (elevator, subway)

<div align="center">
  <img src="images/beijing_dataset_structure.png" width="600"/>
  <p><em>Dataset structure overview.</em></p>
</div>

---

### 🧹 Data Cleaning & Preparation

From the raw dataset (318,851 rows), the following steps were taken:

#### ✔ Selected Relevant Variables  
(id, tradeTime, totalPrice, price, square, rooms, floor, communityAverage, elevator, subway, buildingType, buildingStructure)

#### ✔ Converted Categorical Variables to Factors  
Based on dataset documentation:

- **buildingType** → Tower, Bungalow, Plate/Tower, Plate  
- **buildingStructure** → Mixed, Brick/Wood, Steel, Concrete, etc.  
- **elevator** → 0/1  
- **subway** → 0/1  

<div align="center">
  <img src="images/beijing_datatype_before.png" width="600"/>
  <p><em>Original column types.</em></p>
</div>

<div align="center">
  <img src="images/beijing_datatype_after.png" width="600"/>
  <p><em>Cleaned factor levels.</em></p>
</div>

#### ✔ Removed Missing Values  
2,580 rows containing NA were removed.

<div align="center">
  <img src="images/beijing_na_count.png" width="600"/>
  <p><em>NA count before filtering.</em></p>
</div>

#### ✔ Selected Numerical Features for Correlation  
(totalPrice, price, square, rooms, bathRoom, drawingRoom, communityAverage, floor)

<div align="center">
  <img src="images/beijing_numerical_df.png" width="600"/>
  <p><em>Numerical dataset for correlation analysis.</em></p>
</div>

---

### 📊 Exploratory Data Analysis (EDA)

#### 📈 Histograms: Price & Total Price  
Both price per m² and total price show **right-skewed distributions**, with most prices concentrated between 20,000–60,000.

<div align="center">
  <img src="images/beijing_hist_price.png" width="600"/>
  <p><em>Histogram of price per m².</em></p>
</div>

<div align="center">
  <img src="images/beijing_hist_totalprice.png" width="600"/>
  <p><em>Histogram of total price.</em></p>
</div>

---

### 🔗 Correlation Analysis

Strong positive correlations were found among:

- **totalPrice** ↔ price, square, communityAverage  
- **square** ↔ rooms, bathRoom, drawingRoom  

<div align="center">
  <img src="images/beijing_cor_matrix.png" width="600"/>
  <p><em>Correlation matrix.</em></p>
</div>

<div align="center">
  <img src="images/beijing_corrplot.png" width="600"/>
  <p><em>Corrplot visualization.</em></p>
</div>

---

### 📦 Boxplots: Price by Categorical Variables

#### Building Type  
🏆 Bungalows are the most expensive building type.

<div align="center">
  <img src="images/beijing_box_buildingType.png" width="600"/>
  <p><em>Price vs. Building Type.</em></p>
</div>

#### Building Structure  
Steel/Concrete buildings tend to be more expensive.

<div align="center">
  <img src="images/beijing_box_buildingStructure.png" width="600"/>
  <p><em>Price vs. Building Structure.</em></p>
</div>

#### Elevator  
Homes with elevators → significantly higher prices.

<div align="center">
  <img src="images/beijing_box_elevator.png" width="600"/>
  <p><em>Price vs. Elevator.</em></p>
</div>

#### Subway  
Homes near subway stations → higher average prices.

<div align="center">
  <img src="images/beijing_box_subway.png" width="600"/>
  <p><em>Price vs. Subway Access.</em></p>
</div>

---

### 🔍 Scatterplots

#### Price vs Total Price (by building type & structure)

<div align="center">
  <img src="images/beijing_scatter1.png" width="600"/>
  <p><em>Price vs total price by building type.</em></p>
</div>

<div align="center">
  <img src="images/beijing_scatter2.png" width="600"/>
  <p><em>Price vs total price by building structure.</em></p>
</div>

#### Group-based Regression Slopes

<div align="center">
  <img src="images/beijing_scatter_group1.png" width="600"/>
  <p><em>Square vs Price — grouped by building type.</em></p>
</div>

<div align="center">
  <img src="images/beijing_scatter_group2.png" width="600"/>
  <p><em>Square vs Price — grouped by building structure.</em></p>
</div>

---

### 📅 Average Housing Price by Month

Shows seasonal patterns & monthly variation in housing prices.

<div align="center">
  <img src="images/beijing_monthly_avg_price.png" width="600"/>
  <p><em>Monthly average housing prices.</em></p>
</div>

---

### 🧪 Hypothesis Testing

#### **Question 1:**  
Is the sample mean of housing prices equal to 43,549.6?

Result: **Fail to reject H₀**  
➡ Price is statistically similar to the given value.

<div align="center">
  <img src="images/beijing_ttest_q1.png" width="600"/>
</div>

---

#### **Question 2:**  
Is there a difference in price between **Bungalows** vs **Towers**?

Result: **Reject H₀**  
➡ Bungalows significantly more expensive.

<div align="center">
  <img src="images/beijing_ttest_q2.png" width="600"/>
</div>

---

#### **Question 3:**  
Are 2016 prices greater than 2017 prices?

Result: **Reject H₁**  
➡ 2017 prices are significantly higher.

<div align="center">
  <img src="images/beijing_ttest_q3.png" width="600"/>
</div>

---

### 🧠 Key Insights

- Housing prices in Beijing are strongly influenced by **square meters**, **community average price**, and **building structure**.  
- Homes near subways or with elevators have higher valuations.  
- Bungalows are the most expensive housing type.  
- Prices increased **significantly** from 2016 → 2017.  
- All hypotheses were validated through t-tests (one-sample and two-sample).

---

### 🧠 Skills Demonstrated
- Data wrangling and cleaning in R  
- Visualization with ggplot2 and corrplot  
- Statistical hypothesis testing (one-sample, two-sample t-tests)  
- Interpretation of descriptive and inferential statistics  
- Exploratory data analysis workflows  

</details>

</details>

---

## 🐍 Python Projects
<details>
<summary>📗 Click to view Python-based analytics and ML projects</summary>

---

### [P-Project 1: Seoul Bike Sharing Demand Prediction](https://github.com/FANXYBIN/P-Project1-Seoul-Bike-Sharing-Dataset)
<details>
<summary>🗂️ Click to view project details</summary>

This project applied machine learning models to the Seoul Bike Sharing dataset to predict rental demand based on weather and temporal conditions. The analysis aimed to help optimize bike distribution, adjust operations during weather changes, and identify seasonal rental trends.

* **Dataset:** Seoul Bike Sharing Demand Dataset (UCI Machine Learning Repository, 8760 rows × 14 features).  
* **Tools:** Python (pandas, scikit-learn, seaborn, matplotlib, statsmodels).  
* **Techniques:** Data preprocessing, visualization (histogram, scatterplot, line plot, correlation heatmap), and supervised learning (SVM, Gradient Boosting, Random Forest, and Multiple Linear Regression).  
* **Key Findings:**  
  - **Peak demand** at 8 AM and 6 PM (commuting hours).  
  - **Temperature (15 – 30 °C)** positively affects rentals; **rainfall, humidity, and wind speed** reduce them.  
  - **Summer** records the highest rental activity, followed by spring and autumn.  
* **Models Evaluated:**  
  - **SVM:** Accuracy = 78.99%, Precision = 77.67%, Recall = 78.99%, F1 = 77.93%.  
  - **Gradient Boosting:** Accuracy = 79.83%, F1 = 79.20%.  
  - **Random Forest:** Accuracy = 80.02%, F1 = 79.50%.  
  - **Multiple Linear Regression:** R² = 0.473, F-statistic = 609.8 (significant predictors: hour, temperature, humidity, rainfall).  
* **Insights & Recommendations:**  
  - Use forecasts to rebalance bikes before morning/evening peaks.  
  - Adjust staffing and offer promotions during poor-weather periods.  
  - Promote biking events in warm seasons to leverage natural demand growth.

---

### 📈 Sample Visualizations

**Hourly Rentals by Season**  
![Line Plot](images/seoul_lineplot_season.png)

**Correlation Heatmap**  
![Heatmap](images/seoul_heatmap.png)

**Model Performance Comparison**  
![Model Comparison](images/seoul_model_performance.png)

</details>


</details>

---

## 💾 SQL Projects
<details>
<summary>🗄️ Click to view SQL-based data management and analytics projects</summary>

---

### 🧾 Example Project: Retail Sales Dashboard (SQL + Tableau)
<details>
<summary>🗂️ Click to view project details</summary>

Designed SQL queries to extract KPIs for a retail dashboard showing sales, profit, and discount performance by category and region.  
Integrated with Tableau for interactive visualization.  

* **Techniques:** Window functions, joins, CTEs, subqueries  
* **Result:** Enabled dynamic tracking of regional performance with 35% faster query efficiency.

</details>

</details>

---

## 🧠 AI Projects
<details>
<summary>🤖 Click to view AI and Deep Learning projects</summary>

---

### [A-Project 1: Personality Classification & Model Monitoring with Vertex AI](https://github.com/FANXYBIN/A-Project-Personality-Classification-Model-Monitoring-with-Vertex-AI)

<details>
<summary>🗂️ Click to view project details</summary>

This project developed an **AutoML personality classification model** using **Google Vertex AI**, predicting whether users are **introverts or extroverts** based on social and behavioral traits.  
The full pipeline included **dataset creation**, **model training**, **evaluation**, **deployment**, and **automated monitoring** to ensure long-term reliability and fairness.

* **Dataset:** 2,900 records, 8 behavioral and social features (e.g., Post Frequency, Friends Circle Size, Stage Fear).  
* **Tools:** Google Vertex AI AutoML, Cloud Monitoring, Python, JSON logs.  
* **Techniques:** AutoML classification, confidence threshold tuning, model deployment, and drift detection via monitoring jobs.  
* **Goal:** Demonstrate responsible AI deployment with explainability and drift tracking.

---

### 🧱 Model Creation

**1. Dataset Upload**
- Uploaded the CSV dataset to **Vertex AI Datasets** in Google Cloud Storage.  
- Automatically parsed schema with columns such as:
  - `Post_frequency`, `Social_event_attendance`, `Stage_fear`, `Friends_circle_size`, `Time_spent_alone`, `Stress_level`.  
- Labeled the **target column**: `Personality` → *Introvert* or *Extrovert*.  

**2. AutoML Training**
- Used **Vertex AI AutoML Classification** (no-code training).  
- Enabled automatic feature engineering and model tuning.  
- Ran multiple experiments with two confidence thresholds:
  - **0.5** (balanced accuracy vs. coverage)
  - **0.8** (higher precision, lower recall)

**3. Model Evaluation**
- Vertex AI generated precision–recall and confusion matrix visualizations.  
- **Best model:** 0.5 threshold → optimal for user-friendly applications.  
- **Results:**
  - Accuracy: **93–94%**
  - Introvert recall improved from **93% → 94%**
  - Reduced false positives (Introverts misclassified as Extroverts) from **7% → 6%**

<div align="center">
  <img src="images/vertex_training_evaluation.png" alt="Vertex AI AutoML Training Evaluation" width="600"/>
  <p><em>Model evaluation in Vertex AI AutoML showing improved introvert recall and balanced precision.</em></p>
</div>

**4. Test Model**
<div align="center">
  <img src="images/vertex_testing_model.png" alt="Vertex AI Making Model Process" width="600"/>
  <p><em>Model creation process in Vertex AI AutoML showing dataset import and training setup.</em></p>
</div>

---

### 🧩 Feature Attribution (Explainability)

**Top Features (SHAP Analysis)**  
| Feature | Importance | Interpretation |
|----------|-------------|----------------|
| Post_frequency | ⭐ Highest | High posting frequency = Extroversion |
| Social_event_attendance | ⭐⭐ | Social activity drives Extrovert classification |
| Stage_fear | ⭐⭐ | Low stage fear → Extrovert; high fear → Introvert |
| Time_spent_alone | ⭐ | Longer time alone → Introversion |
| Friends_circle_size | ⭐ | Moderate indicator of social confidence |

<div align="center">
  <img src="images/vertex_feature_attribution.png" alt="Feature Attribution Plot" width="600"/>
</div>

---

### 🧩 Model Monitoring Configuration

**Monitoring Components**
- **Input Drift:** Detects distribution shifts in features.  
- **Prediction Drift:** Identifies output pattern shifts over time.  
- **Attribution Drift:** Detects evolving feature importance patterns.  

**Implementation**
- Configured via **Vertex AI Monitoring**:
  - Drift threshold = 0.1  
  - Enabled **email alerts** on drift detection  
  - Sampling rate: 100% (all predictions logged)  
- Used **Google Cloud Shell** commands to update monitoring jobs and log prediction drift in JSON format.  

<div align="center">
  <img src="images/vertex_monitoring_overview.png" alt="Vertex AI Monitoring Overview" width="600"/>
</div>

---

### 🎯 Key Takeaways
- **AutoML enabled fast and accessible modeling**, ideal for small teams.  
- **Model monitoring ensured trustworthiness** by detecting data shifts early.  
- **Explainability (SHAP)** supported responsible AI interpretation.  
- Framework applicable to social platforms, HR analytics, and CRM recommendation systems.  

---

### 🧠 Skills Demonstrated
- Google Vertex AI AutoML training, tuning, and deployment  
- Feature attribution and explainable AI (SHAP)  
- Model drift detection and governance  
- MLOps monitoring configuration with Google Cloud  


</details>


### [A-Project 2: NLTK Text Analysis & Classification — *Alice in Wonderland*](https://github.com/FANXYBIN/A-Project-2-NLTK-Text-Analysis-Classification)

<details>
<summary>🗂️ Click to view project details</summary>

This project applied **Natural Language Processing (NLP)** techniques using **NLTK** and **scikit-learn** to analyze and classify text from *Alice’s Adventures in Wonderland* by Lewis Carroll.  
The goal was to explore linguistic patterns and train a model to classify chapters based on word usage and stylistic features.

* **Dataset:** *Alice’s Adventures in Wonderland* (Project Gutenberg)  
* **Tools:** Python, NLTK, scikit-learn, CountVectorizer, matplotlib  
* **Techniques:** Tokenization, stopword removal, lemmatization, vectorization, Naive Bayes classification  
* **Goal:** Identify vocabulary trends across chapters and evaluate classification accuracy.

---

### ⚙️ Text Preprocessing
1. **Download text:** Retrieved from Project Gutenberg using `requests`.  
2. **Clean content:** Removed special characters and headers.  
3. **Split into chapters:** Used regex to identify Roman numeral chapter headings.  
4. **Tokenize & remove stopwords:** NLTK’s `word_tokenize()` and `stopwords`.  
5. **Vectorize:** Converted text into numerical features using `CountVectorizer`.

---

### 🧩 Vocabulary Construction
Custom stopword lists were merged and applied to extract the most representative words.  
Each token received a unique index in the vocabulary using **CountVectorizer**.

<div align="center">
  <img src="images/NLTK_stopword lists.png" alt="Vocabulary construction code" width="500"/>
  <p><em>Vocabulary generation with token–ID mapping for words in the corpus.</em></p>
</div>

---

### 🧪 Classification Experiment
Each chapter was treated as a labeled text sample.  
A **Multinomial Naive Bayes** classifier was trained to predict which chapter a given excerpt belonged to.

**Process:**
- Converted chapters to numerical vectors.  
- Split dataset into **train (70%)** and **test (30%)** sets.  
- Trained the model and evaluated prediction accuracy.

<div align="center">
  <img src="images/NLTK_report.png" alt="Model training and evaluation" width="600"/>
  <p><em>Model training and classification report with 0.375 accuracy.</em></p>
</div>

| Metric | Value |
|---------|-------|
| Accuracy | 0.38 |
| Macro Avg Precision | 0.25 |
| Macro Avg Recall | 0.30 |
| Weighted Avg F1 | 0.33 |

Despite modest performance, the model captured stylistic variation between chapters—such as vocabulary density and dialogue frequency.

---

### 📊 Linguistic Insights
- Frequent nouns: **Alice, Queen, King, Rabbit, Time**  
- Frequent verbs: **said, thought, went, replied**  
- Dialogue-heavy chapters contain more pronouns and verbs.  
- Later chapters emphasize descriptive adjectives and nouns.

---

### 🧠 Skills Demonstrated
- Text preprocessing and tokenization using **NLTK**  
- Feature extraction with **CountVectorizer**  
- Supervised text classification using **Naive Bayes**  
- Evaluation and linguistic interpretation of text patterns  

📓 [View Jupyter Notebook](codes/NLTK_Alice.ipynb)


</details>


### [A-Project 3: Amazon Reviews Sentiment Analysis Microservice](https://github.com/FANXYBIN/A-Project3-Amazon-Reviews-Sentiment-Analysis-Microservice)

<details>
<summary>🗂️ Click to view project details</summary>

This project developed a **binary sentiment classification model** for Amazon product reviews and deployed it as a **serverless microservice** using Docker and Google Cloud Run.  
Multiple classical and deep learning models were evaluated, including Logistic Regression, Linear SVM, LSTM, and DistilBERT.

* **Dataset:** Amazon Reviews Dataset (Kaggle, ~4M reviews)  
* **Tools:** Python, scikit-learn, TensorFlow, Hugging Face Transformers, Docker, Google Cloud Run  
* **Techniques:** Text preprocessing, TF–IDF vectorization, LSTM architecture, DistilBERT fine-tuning, microservice deployment  
* **Goal:** Build a production-ready sentiment classification API with high accuracy and scalable inference.

---

### ⚙️ Data Preparation
1. Merged Kaggle train & test sets (~4M rows).  
2. Removed 231 missing rows.  
3. Re-labeled sentiment: **0 = negative, 1 = positive**.  
4. Sampled **1 million** reviews for efficient training.  
5. Applied text cleaning: lowercase, remove punctuation, digits, and extra whitespace.  
6. Created `clean_text` feature for modeling.

---

### 🧩 Model Development
Four models were trained and evaluated on the prepared dataset.

| Model | Accuracy | Precision | Recall | F1 | Notes |
|-------|-----------|------------|---------|--------|--------|
| Logistic Regression | 0.87 | 0.87 | 0.87 | 0.87 | Classical baseline |
| Linear SVM | 0.867 | 0.87 | 0.87 | 0.87 | Strong TF–IDF performance |
| LSTM | 0.907 | 0.92 | 0.90 | 0.91 | Captures sequential patterns |
| **DistilBERT** | **0.949** | **0.96** | **0.94** | **0.95** | Best-performing model |

---

### ☁️ Model Deployment (Google Cloud Run)
The final DistilBERT model was deployed as a **serverless microservice**.

**Deployment Steps**
- Exported model + tokenizer  
- Saved artifacts in Google Cloud Storage  
- Built Docker container for Flask API  
- Pushed and deployed container to Cloud Run  


---

### 🧪 API Testing
<div align="center"> <img src="images/API_testing-postman.png" alt="API Testing Screenshot 1" width="600"/> <p><em>Testing the API with Postman.</em></p> </div>
<div align="center"> <img src="images/API_testing-google colab.png" alt="API Testing Screenshot 2" width="600"/> <p><em>Testing the API with Google Colab.</em></p> </div>

---

### 💡 Applications

1.Real-time customer review analysis.

2.Automated support message classification.

3.E-commerce product sentiment trends.

4.Social media & brand monitoring.

---

### 🧠 Skills Demonstrated

1.Sentiment analysis with classical ML, LSTM, and Transformers.

2.Deep learning model comparison.

3.Docker containerization.

4.Google Cloud Run deployment.

5.REST API development for ML services.

</details>


### [A-Project 4: Environmental Sound Classification (ESC-50) with ResNet-50](https://github.com/FANXYBIN/A-Project4-Environmental-Sound-Classification-ESC-50-with-ResNet-50)

<details>
<summary>🗂️ Click to view project details</summary>

This project explored **environmental sound classification** using the **ESC-50 dataset**, converting audio clips into **Mel spectrograms** and training a fine-tuned **ResNet-50 CNN model**.  
The workflow included data preprocessing, spectrogram generation, augmentation (SpecAugment), model training, and evaluation with accuracy and confusion matrix metrics.

* **Dataset:** ESC-50 (2,000 labeled audio clips, 50 sound classes)  
* **Tools:** Python, PyTorch, torchaudio, librosa, matplotlib  
* **Techniques:** Mel spectrograms, SpecAugment, custom dataset loader, transfer learning (ResNet-50)  
* **Goal:** Build an effective audio classifier by transforming audio into images for CNN-based learning.

---

### 🎼 Dataset & Preprocessing

- ESC-50 consists of **2,000 WAV files** across **50 categories** (animals, natural sounds, human noises, etc.).  
- Converted each WAV file into a **Mel spectrogram** using `librosa`, then saved as `.png` images for image-based CNN training.  
- Implemented a **custom PyTorch dataset class (SpecDataset)** to load spectrograms and labels.  
- Applied preprocessing: resizing, normalization, and **SpecAugment** (frequency & time masking).  
- Split dataset into **train / validation / test** using filename rules.

<div align="center">
  <img src="images/esc50_spectrogram_example.png" width="600"/>
  <p><em>ESC-5 Dataset.</em></p>
</div>
<div align="center">
  <img src="images/Mel Spectrogram.png" width="600"/>
  <p><em>Mel Spectrogram with frequency & time masking (SpecAugment).</em></p>
</div>

---

### 🏗️ Model Architecture & Training

- Used **ResNet-50 pretrained on ImageNet**, freezing all layers except:  
  - Layer 3  
  - Layer 4  
  - Final classifier  
- Replaced the last fully connected layer with a **50-class classifier**.  
- Set learning rate = **0.01**, with **LR decay (×0.5 every 6 epochs)**.  
- Training configured for **66 epochs** with **early stopping (patience = 6)**.  
- Saved checkpoint whenever **validation loss improved**.

<div align="center">
  <img src="images/esc50_training_curve.png" width="600"/>
  <p><em>Training vs. validation loss and accuracy trends.</em></p>
</div>

**Observations:**
- Training accuracy → **~99%**, Validation accuracy → **~80%**  
- Validation loss stabilized after ~18 epochs  
- Training–validation accuracy gap indicates **mild overfitting**, improved by SpecAugment

---

### 📊 Evaluation

- Training stopped at **epoch 41** due to early stopping  
- **Best validation accuracy:** **80.34%**  
- **Validation loss:** **0.5999**  
- **Test accuracy:** **70.94%**  
- **Test loss:** **1.0581**



Performance was strong on distinct classes (e.g., dog bark, rain) but more challenging for acoustically similar categories (e.g., engine sounds vs. machinery).

---

### ❓ Q&A

**1. Why this dataset?**  
ESC-50 is widely used for environmental sound research. It is balanced, cleanly labeled, and manageable for academic deep-learning workflows.

**2. What modifications were needed?**  
- Converted WAV files → Mel spectrogram images  
- Created a **PyTorch Dataset** to load spectrograms  
- Designed filename-based train/val/test splitting  
- Implemented preprocessing & SpecAugment  
- Built custom classifier for a 50-class CNN task

**3. What challenges did you encounter?**  
- Inconsistent filename patterns → solved using **regex**  
- Overfitting → mitigated using **SpecAugment + dropout**  
- Normalization distorted spectrogram visuals → fixed by debugging preview pipeline  
- Balancing training time vs. performance required LR tuning

**4. Would the model be deployable? Why or why not?**  
Not yet. While achieving strong accuracy (80% val, 71% test), deployment-readiness requires:  
- Real-time spectrogram generation  
- Input-output interface (microservice / app)  
- Model size optimization  
- Handling unseen audio conditions  

---

### 🧠 Skills Demonstrated

- Audio → image transformation using Mel spectrograms  
- CNN fine-tuning (ResNet-50) for audio tasks  
- PyTorch training pipelines, LR scheduling, early stopping  
- Data augmentation via SpecAugment  
- Model evaluation with confusion matrix  

</details>




### [A-Project 5: CIFAR-10 Image Classification & Overfitting Analysis (ResNet-34 + fastai)](https://github.com/FANXYBIN/A-Project-5-CIFAR-10-Image-Classification-Overfitting-Analysis-ResNet-34-fastai-)

<details>
<summary>🗂️ Click to view project details</summary>

This project explored **overtraining/overfitting** in deep learning using the **CIFAR-10** dataset and a **ResNet-34** model trained with **fastai**.  
By intentionally causing overtraining and then applying prevention techniques, the project demonstrates how training dynamics affect generalization.

* **Dataset:** CIFAR-10 (60,000 images, 10 classes)  
* **Tools:** Python, fastai, PyTorch, torchvision  
* **Techniques:** Transfer learning, fine-tuning, LR finder, early stopping, data augmentation  
* **Goal:** Identify overfitting and apply methods to improve generalization.

<div align="center">
  <img src="images/cifar10_dataset.png" width="650"/>
  <p><em>CIFAR10 Dataset Sample.</em></p>
</div>

---

### ⚙️ Model Setup & Training Procedure

1. Loaded CIFAR-10 and created a **fastai Learner** with a **pretrained ResNet-34**.  
2. Tracked **error rate** and trained for **2 epochs**, achieving strong initial performance.  
3. Examined predictions → most predictions matched ground truth correctly.  
4. Viewed **activation outputs** (10-dim probability vectors).  
5. Experimented with different learning rates:
   - **Too high LR (0.1)** → training loss spiked to **2.73**, showing unstable learning.  
   - Used **LR Finder** → optimal LR ≈ **4.7e-4**.  
6. Trained randomly initialized layers for **3 epochs** using `fit_one_cycle`.  
7. Fine-tuned entire model for **6–12 more epochs** with LR range **1e-5 to 1e-4**.

<div align="center">
  <img src="images/cifar10_training_curve.png" width="650"/>
  <p><em>Training vs. validation loss curve showing early signs of overfitting.</em></p>
</div>

---

### 📉 Detecting Overfitting

According to training logs :contentReference[oaicite:1]{index=1}:

- With frozen layers → validation loss dropped from **0.37 → 0.18**, error rate from **8.9% → 6.4%**.  
- After unfreezing → training & validation loss continued improving until **epoch 8**.  
- Beyond epoch 8 →  
  - Validation loss began to **fluctuate**  
  - Error rate slightly **increased**  
  - Training loss kept decreasing → **clear overtraining**  

💡 **Conclusion:** The model reached optimal performance at **epoch 8**. Training past that point caused mild overfitting.

---

### 🛠️ Methods to Prevent Overtraining

From the assignment discussion :contentReference[oaicite:2]{index=2}, the following techniques helped reduce overfitting:

#### ✔ Early Stopping  
Stops training when validation loss stops improving — prevents memorization of training data.

#### ✔ Data Augmentation  
Transforms such as:
- Rotate  
- Crop  
- Flip  
- Color jitter  
Increase data diversity → improves generalization.

#### ✔ L2 Weight Regularization (Weight Decay)  
Encourages smaller weights → reduces model complexity → prevents overfitting.

---

### 🧪 Class-wise Performance (Confusion Matrix)

The CIFAR-10 confusion matrix showed:  
**Best-performing classes**
- 🛳️ Ship — **977/1000 correct**  
- 🚗 Automobile — **973/1000 correct**  
- 🐸 Frog — **973/1000 correct**

**Worst-performing classes**
- 🐱 Cat — **884/1000**  
- 🚚 Truck — **858/1000**  
- 🐶 Dog — **933/1000**

<div align="center">
  <img src="images/cifar10_confusion_matrix.png" width="650"/>
  <p><em>Confusion matrix showing class-wise model accuracy.</em></p>
</div>

🔍 **Why these classes struggled:**  
They are visually similar to other classes:
- Cats ↔ Dogs  
- Trucks ↔ Automobiles  

### 🔧 Recommendation for Improvement
- Apply **stronger augmentation** specifically for similar classes  
- Use **deeper models** (ResNet-50, EfficientNet) for better feature extraction  
- Use **class-specific fine-tuning** or **ensemble models** to separate difficult class pairs

---

### 🧠 Skills Demonstrated
- Transfer learning with PyTorch + fastai  
- Training dynamics analysis  
- Detecting overtraining using loss & error rate  
- Using LR finder and one-cycle policy  
- Confusion matrix evaluation  
- Applying regularization and augmentation

</details>

</details>

---

## 📊 Dashboard Projects

<details>
<summary>🖥️ Click to view Dashboard projects</summary>

---

### [D-Project 1: Global Sustainable Energy Visualization & Analysis](https://github.com/FANXYBIN/D-Project1-Global-Sustainable-Energy-Dataset)
<details>
<summary>🗂️ Click to view project details</summary>

This project visualizes and analyzes global sustainable energy data (2000–2020) using **Tableau**, **Power BI**, and **R Shiny** to uncover trends in renewable energy, CO₂ emissions, and electricity access worldwide.

* **Dataset:** *Global Data on Sustainable Energy* (Kaggle, 3,649 rows × 21 features).  
* **Tools:** Tableau, Power BI, R Shiny (R, ggplot2, dplyr, shinydashboard), DAX.  
* **Techniques:** Data cleaning, parameter-based filtering, interactive dashboard design, and regression visualization.  
* **Objective:** Explore the transition toward renewable energy and identify disparities in global access to electricity.

---

### 📊 Dashboards & Insights

**Tableau Dashboard**
- Explored access to electricity, energy generation by source, and renewable growth across years.  
- Used maps, bar charts, and parameters (Top X) to identify top-performing countries.  
- Highlighted a steady increase in renewable electricity generation and energy aid to developing countries.
 
<img src="images/Global%20Sustainable%20Energy_Tableau.png" width="1000"/>

**Power BI Dashboard**
- Designed “Global Energy Development Indicators” with slicers, cards, maps, and line charts.  
- Created DAX measures to calculate renewable, nuclear, and fossil fuel shares.  
- Revealed that renewable energy share is rising while fossil fuel dependence remains high.

<img src="images/Global%20Sustainable%20Energy_PBI.png" width="700"/>

**R Shiny Dashboard**
- Built an interactive web app with filters for **year** and **country**.  
- Visualized:
  - Renewable electricity capacity growth (line chart).  
  - Energy generation mix (stacked bar).  
  - CO₂ vs. low-carbon electricity (scatter with regression).  
  - Top 10 CO₂-emitting countries (bar chart).  
- Demonstrated negative correlation between CO₂ emissions and low-carbon electricity share.

<img src="images/Global%20Sustainable%20Energy_RShiny.png" width="700"/>

---

### 🌱 Key Findings
- Renewable energy generation increased steadily between 2000–2020.  
- Developing countries benefited from financial aid but still rely heavily on fossil fuels.  
- Low-carbon electricity adoption significantly reduces CO₂ emissions.  
- Africa and South Asia show persistent electricity access gaps.

---

### 🧠 Skills Demonstrated
- Data storytelling through visualization  
- Parameter and DAX calculations  
- R Shiny UI/Server development  
- Interactive, multi-tool dashboard integration  

**[Dataset Source → Kaggle](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy/data)**  
**[R Shiny Reference → Appsilon Blog](https://www.appsilon.com/post/r-shiny-in-life-sciences-examples)**  

</details>

### [D-Project 2: PowerTrust Renewable Energy Dashboard](https://github.com/FANXYBIN/D-Project2-PT-Renewable-Energy-Dashboard)
<details>
<summary>🗂️ Click to view project details</summary>

This project was completed in collaboration with **PowerTrust**, focusing on developing a Tableau dashboard to visualize renewable energy generation and Distributed Renewable Energy Certificates (D-RECs) across multiple countries.  
The dashboard helps PowerTrust identify high-performing projects, track emission reductions, and make data-driven sustainability decisions.

* **Dataset:** PowerTrust Renewable Energy Dataset (12,432 entries across 13 countries).  
* **Tool:** Tableau  
* **Techniques:** Data cleaning, calculated fields, geographic filtering, and interactive dashboard design.  
* **Key Objectives:**  
  - Visualize renewable project performance by country and developer.  
  - Track CO₂ reduction and D-REC generation.  
  - Identify outliers, anomalies, and operational improvement areas.  

---

### 📊 Dashboard Highlights

**Global Dashboard**
- Interactive map visualizing project distribution and energy generation.  
- Summary cards displaying total energy, CO₂ reduction, and D-RECs.  
- Filters for country, developer, and site type for dynamic exploration.  

**Country-Level Dashboards**
- **India:** 558 projects, 8.2B g/kWh CO₂ reduced, 13,281 D-RECs.  
- **Ghana:** 3 projects with 8.1B g/kWh CO₂ reduced and 13,077 D-RECs.  
- **Vietnam:** Steady performance with strong emission reductions across projects.  

**Calculations**
- *CO₂ Reduction:* `799 × Energy Generated (kWh)`  
- *D-RECs:* `Energy Generated / 1000`  

---

### 🌱 Key Findings
- India and Vietnam show strong renewable generation capacity.  
- Ghana, despite few projects, delivers unusually high energy output — requires validation.  
- Some projects have mismatched SMR start/end dates, corrected through calculated fields.  
- Underperforming countries (e.g., Libya, Algeria) indicate opportunities for expansion.  

---

### 🔍 Recommendations
- Review data anomalies by consulting developers.  
- Integrate private APIs (e.g., DHI, DNI, GHI) for more precise solar metrics.  
- Incorporate SDG metrics (via SDG Action Manager) to track local sustainability impact.  

---

### 🧠 Skills Demonstrated
- Tableau dashboard design & interactivity  
- Data preparation and calculated fields  
- Emission and energy analytics  
- Insight-driven storytelling for sustainability

---

### 🖥️ Dashboard

<img src="images/PT%20Dashboard.png" width="1000"/>
<img src="images/PT%20Dashboard_India.png" width="1000"/>
<img src="images/PT%20Dashboard_India2.png" width="1000"/>

</details>


### [D-Project 3: U.S. Traffic Accident Analysis Dashboard](https://github.com/FANXYBIN/D-Project3-U.S.-Traffic-Accident-Analysis-Dashboard)
<details>
<summary>🗂️ Click to view project details</summary>

This project analyzed a large-scale dataset of U.S. traffic accidents from **2016–2023**, containing over **7.7 million records** and 46 features across all 49 states.  
Using **PySpark** for preprocessing and **Tableau** for visualization, we developed a scalable analytical dashboard to identify trends, high-risk regions, and contributing factors behind road accidents.

* **Dataset:** U.S. Traffic Accident Dataset (2016–2023, 7.7M records, 46 features).  
* **Tools:** PySpark, Tableau.  
* **Techniques:** Data reduction, feature selection, distributed computing, and interactive dashboard design.  
* **Objective:** Provide policymakers with actionable insights to enhance traffic safety and reduce accidents.  

---

### ⚙️ Data Preparation
- Used **PySpark** to handle large parquet files efficiently (up to 70GB).  
- Compared Pandas vs. PySpark performance — PySpark achieved stable runtime and avoided memory crashes.  
- Selected **14 essential features** (e.g., Severity, Start Time, Weather Condition, Traffic Feature) for focused analysis.  
- Reduced data size from **3.06 GB → 1.53 GB**, improving Tableau performance and stability.  

---

### 📊 Dashboard Components
- **Slicers:** Filter by year and severity to compare accident trends (Current Year vs. Previous Year).  
- **Cards:** Display monthly trends in accidents by severity level.  
- **Map:** Visualizes accident density by severity using color codes (blue–green–orange–red).  
- **Donut Chart:** Shows weather condition proportions during accidents.  
- **Stacked Bar Chart:** Highlights accident frequency near key traffic features (e.g., crossings, junctions, signals).  

<img src="images/US%20Dashboard1.png" width="700"/>
<img src="images/US%20Dashboard2.png" width="700"/>
<img src="images/US%20Dashboard3.png" width="700"/>

---

### 🚦 Key Findings
- **Steady Increase in Accidents:** +74.79% (2016→2017), +24.42% (2017→2018).  
- **Severity 2 Accidents Dominate:** Most common category across all years.  
- **Weather:** ~48% of accidents occurred in clear weather, showing human/infrastructure factors are major causes.  
- **Traffic Signals:** The most common accident location (≈21%).  
- **Urban Hotspots:** Accidents concentrated near major traffic infrastructures and densely populated areas.  

---

### 🔍 Recommendations
- Strengthen driver education and compliance near intersections.  
- Improve signal visibility and timing to reduce signal-related crashes.  
- Use predictive analytics on historical data to allocate enforcement and maintenance resources efficiently.  

---

### 🧠 Skills Demonstrated
- Big data preprocessing with **PySpark**  
- Interactive visualization using **Tableau**  
- Feature selection and data reduction for scalability  
- Analytical storytelling for transportation safety insights  

</details>

### [D-Project 4: OpenRep Social Media Analytics Dashboard](https://github.com/FANXYBIN/D-Project4-OP-Social-Media-Analytics-Dashboard)

<details>
<summary>🗂️ Click to view project details</summary>

This capstone project, in collaboration with **OpenRep**, focused on designing a full-stack **data pipeline and analytics dashboard** to evaluate social media engagement across multiple platforms — including Instagram, Twitter(X), LinkedIn, Pinterest, and Facebook.  
The system automated data ingestion, cleaning, anomaly detection, and visualization to support OpenRep’s content performance insights.

* **Dataset:** Multi-platform social media data (Instagram, X, Facebook, LinkedIn, Pinterest).  
* **Tools:** Python (Pandas, Plotly), R (ggplot2, dplyr), Tableau, Power BI.  
* **Techniques:** API data extraction, automated preprocessing, EDA, anomaly detection, KPI formulation, and dashboard visualization.  
* **Goal:** Create a centralized dashboard integrating platform-specific metrics to identify engagement trends and detect anomalies.

---

### ⚙️ Pipeline Overview

**1. API Development**
- Automated API converts platform data into structured Excel files.  
- Removed redundant columns, standardized field names, and validated schema consistency.  
- Generated seven cleaned datasets:
  - `fact_facebook`, `fact_instagram`, `fact_linkedin`, `fact_pinterest`, `fact_twitter`, `fact_gmb`, and `dim_post`.  

**2. Data Cleaning**
- Filled missing values using median imputation.  
- Dropped columns with >80% missing values.  
- Flagged residual errors for manual review.

**3. EDA**
- **Instagram:** Engagement, followers, impressions, and reach were strongly correlated.  
- **Twitter(X):** Clear anomaly clusters in July 2024 and January 2024; engagement spikes >850 followed by steep declines.  
- **Pinterest:** Lower but stable engagement patterns.  
- **Facebook:** High variance and isolated outlier peaks.  
- **LinkedIn:** Consistent moderate engagement rate and audience growth.

---

### 🔎 Post-Perspective Analytics

**KPI Formulas**
- *Engagement Rate* = Engagement ÷ Reach  
- *Like Rate* = Likes ÷ Reach  
- *Impressions per User* = Impressions ÷ Reach  

**Insights**
- Instagram and LinkedIn maintain consistent engagement and like rates.  
- Facebook occasionally yields top-performing viral posts.  
- Twitter(X) and Pinterest show wider metric dispersion, indicating platform volatility.  
- Facebook: strong re-exposure → brand recall;  
  Instagram + LinkedIn: balanced delivery;  
  Twitter + Pinterest: strong user acquisition but lower retention.

---

### 📊 Dashboard Framework

<div align="center">
  <img src="images/OP_Overview.png" alt="OpenRep Analytics Dashboard Overview" width="600"/>
</div>

- **Anomaly Detection:** Automatically flags metric spikes/drops (engagement, impressions, profile views).  
- **Custom Filters:** Users can select *platform* and *metric* (e.g., profile views, engagement rate, CTR, follower growth).  
- **Visualization Layers:**  
  - Platform-level engagement tracking  
  - Post-level performance analysis  
  - Cross-platform KPI comparisons  

<div align="center">
  <img src="images/OP_Instagram.png" width="450"/>
  <img src="images/OP_X.png" width="480"/>
</div>

<div align="center">
  <img src="images/OP_linkedin.png" width="450"/>
  <img src="images/OP_pinterest.png" width="450"/>
</div>

---

### 📈 Key Findings
- Instagram engagement rose steadily with clear seasonal anomalies (Mar 2024 & Mar 2025).  
- Twitter(X) recorded spikes in engagement and impressions but could not sustain post-peak activity.  
- LinkedIn maintained the most stable engagement rate across 2024–2025.  
- Post-level performance confirmed campaign-specific patterns (e.g., **Campaign-2819c4db-452c-43c3-8bec-267af48dcf41**).  

---

### 🧠 Skills Demonstrated
- Multi-platform API integration  
- Data cleaning and pipeline automation  
- Anomaly detection (EDA-driven)  
- KPI engineering and visualization design  
- Cross-team collaboration & presentation (MIT Sloan reference integration)  

</details>

</details>

---

## 👤 About Me
Hi, I’m **James Li**, a data analyst and aspiring AI practitioner.  
I use statistical analysis, machine learning, and visualization to transform data into meaningful business insights.

📫 **Connect with me:**  
- [LinkedIn](https://linkedin.com/in/xuanbin-li)  
- [GitHub](https://github.com/FANXYBIN)  
- ✉️ Email: james.xb.li13@gmail.com  
