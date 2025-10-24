# 🌟 Xuanbin Li — Data Analytics & AI Portfolio

Welcome to my project portfolio!  
I specialize in data analytics, statistical modeling, and AI-driven insights using **R**, **Python**, **SQL**, **AI**, and **Dashboard**.  
Click each section below to explore projects by language or domain.

---

## 🧮 R Projects
<details>
<summary>📘 Click to view R-based projects</summary>

---

### [R-Project 1: Superstore Sales Performance Analysis & Forecasting](https://github.com/FANXYBIN/Project1_Superstore_Dataset)

<details>
<summary>🗂️ Click to view project details</summary>

This project analyzed a global superstore dataset (2011–2014) using R to explore business performance and forecast future sales and profit trends.

* **Dataset:** Superstore dataset from Kaggle (2011–2014), containing sales, profit, discounts, and shipping details across multiple regions and categories.  
* **Tools:** R (tidyverse, forecast, corrplot, treemap, data.table)  
* **Techniques:** Data cleaning, visualization (bar, pie, scatter, box, treemap), correlation analysis, and ARIMA time series forecasting.  
* **Key Insights:**  
  - APAC market and Central region achieved the highest sales and profit.  
  - "Phones" under *Technology* had the highest sales, while *Tables* incurred losses.  
  - Profit negatively correlated with Discount.  
  - ARIMA forecast predicted a continued increase in 2015 sales and profit.  
* **Result:** Provided data-driven insights into regional and category-level performance and built an ARIMA model for forecasting next-year trends.  

---

### 📈 Sample Visualizations

**ARIMA Forecast for Next Year (Profit)**  
![](images/ARIMA%20forecast%20for%20Profit.png)

**ARIMA Forecast for Next Year (Sales)**  
![](images/ARIMA%20forecast%20for%20Sales.png)

</details>

---

### [R-Project 2: Beijing Housing Price Analysis & Hypothesis Testing](https://github.com/FANXYBIN/Project2_Beijing_Housing_Price_Dataset)
<details>
<summary>🗂️ Click to view project details</summary>

This project analyzes housing prices in Beijing using R. The goal was to understand key factors influencing house prices and test hypotheses about housing market trends between 2016 and 2017.

* **Dataset:** Housing Price in Beijing dataset from Kaggle (318,851 observations, 26 features).  
* **Tools:** R (tidyverse, ggplot2, corrplot, dplyr, stats).  
* **Techniques:** Data cleaning, visualization (histograms, scatterplots, boxplots, correlation plots), and hypothesis testing (one-sample and two-sample t-tests).  
* **Key Insights:**  
  - Total price strongly correlated with house area, community average, and number of rooms.  
  - Houses near subways or with elevators tend to have higher average prices.  
  - “Bungalows” are the most expensive building type, while “Towers” are more affordable.  
  - Average housing prices increased significantly from 2016 to 2017.  
* **Result:** Provided data-driven insights into how structural and locational factors affect housing prices in Beijing and validated findings through statistical hypothesis testing.

---

### 📈 Sample Visualizations

**Boxplots: Price vs Building Type & Structure**  
![Boxplot Building Type](images/boxplot_buildingtype.png)  
![Boxplot Building Structure](images/boxplot_buildingstructure.png)

**Correlation Among Key Variables**  
![Correlation Plot](images/corrplot_features.png)

**Average Monthly Housing Price**  
![Average Price by Month](images/avg_price_by_month.png)

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

### 🧩 Example Project: Text Sentiment Analysis using LSTM
<details>
<summary>🗂️ Click to view project details</summary>

Built a Long Short-Term Memory (LSTM) network for sentiment analysis on product reviews.  

* **Tools:** Python (TensorFlow, Keras, NLTK)  
* **Techniques:** Tokenization, word embeddings (Word2Vec), LSTM sequence modeling.  
* **Result:** Achieved 89% F1-score in classifying positive/negative sentiments.  

---

### 📈 Sample Visualizations
![Training Accuracy](images/ai_lstm_accuracy.png)
![Loss Curve](images/ai_lstm_loss.png)

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
<summary>📊 Click to view project details</summary>

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

</details>

---

## 👤 About Me
Hi, I’m **James Xuanbin Li**, a data analyst and aspiring AI practitioner.  
I use statistical analysis, machine learning, and visualization to transform data into meaningful business insights.

📫 **Connect with me:**  
- [LinkedIn](https://linkedin.com/in/xuanbin-li)  
- [GitHub](https://github.com/FANXYBIN)  
- ✉️ Email: james.xb.li13@gmail.com  
