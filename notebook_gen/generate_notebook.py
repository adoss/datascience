from nbformat import v4 as nbf
from datetime import datetime

cells = []

# Title & Intro
cells.append(nbf.new_markdown_cell("# Customer Segmentation Project - Full Code Version"))
cells.append(nbf.new_markdown_cell(f"_Date: {datetime.today().strftime('%Y-%m-%d')}_"))
cells.append(nbf.new_markdown_cell("## 1. Problem Statement\nUnderstand customer behaviors to build marketing segments and drive business value."))

# Load and explore data
cells.append(nbf.new_markdown_cell("## 2. Load and Explore Data"))
cells.append(nbf.new_code_cell("""import pandas as pd

# Load data
df = pd.read_csv("your_data.csv")
df.head()"""))
cells.append(nbf.new_code_cell("""# Basic structure
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()"""))

# Feature engineering
cells.append(nbf.new_markdown_cell("## 3. Data Cleaning and Feature Engineering"))
cells.append(nbf.new_code_cell("""# Convert Dt_Customer to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Customer_Tenure'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days

# Create age and total spending features
df['Age'] = 2025 - df['Year_Birth']
df['Children'] = df['Kidhome'] + df['Teenhome']
df['Total_Spend'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)"""))

# EDA
cells.append(nbf.new_markdown_cell("## 4. Exploratory Data Analysis (EDA)"))
cells.append(nbf.new_markdown_cell("### Univariate Analysis"))
cells.append(nbf.new_code_cell("""import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Income'], kde=True)
plt.title("Income Distribution")
plt.show()"""))
cells.append(nbf.new_markdown_cell("### Bivariate Analysis"))
cells.append(nbf.new_code_cell("""sns.boxplot(x='Education', y='MntGoldProds', data=df)
plt.title("Gold Product Spend by Education Level")
plt.show()"""))

# Preprocessing
cells.append(nbf.new_markdown_cell("## 5. Preprocessing for Clustering"))
cells.append(nbf.new_code_cell("""from sklearn.preprocessing import StandardScaler

features = ['Income', 'Recency', 'Customer_Tenure', 'Age', 'Children', 'Total_Spend']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)"""))

# KMeans clustering
cells.append(nbf.new_markdown_cell("## 6. Clustering - KMeans"))
cells.append(nbf.new_code_cell("""from sklearn.cluster import KMeans

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Curve")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()"""))
cells.append(nbf.new_code_cell("""from sklearn.metrics import silhouette_score

for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score for k={k}: {score:.3f}")"""))
cells.append(nbf.new_code_cell("""# Final Clustering
final_kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)"""))

# Cluster profiling
cells.append(nbf.new_markdown_cell("## 7. Cluster Profiling"))
cells.append(nbf.new_code_cell("""df.groupby('Cluster')[features].mean().T"""))

# Insights & Recommendations
cells.append(nbf.new_markdown_cell("## 8. Insights and Business Recommendations"))
cells.append(nbf.new_markdown_cell("""
- **Cluster 0:** Young low-spenders — target with promotions.
- **Cluster 1:** Affluent loyalists — offer exclusive perks.
- **Cluster 2:** Online-focused — optimize digital channels.
- **Cluster 3:** Price-sensitive — leverage discount campaigns.
"""))

# Create the notebook
notebook = nbf.new_notebook(cells=cells)

with open("Customer_Segmentation_FullCode.ipynb", "w") as f:
    f.write(nbf.writes(notebook))
print("Notebook generated: Customer_Segmentation_FullCode.ipynb")
