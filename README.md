# Customer Segmentation Using K-Means Clustering

This project demonstrates customer segmentation using the K-Means clustering algorithm on the Mall Customers dataset. The goal is to group customers into distinct segments based on their demographic and spending behavior, enabling targeted marketing strategies and business insights.

## Dataset

- **File:** `Mall_Customers.csv`
- **Description:** Contains data for 200 customers of a mall, including features such as CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).

## Project Structure

- `customer_segmentation.ipynb`: Jupyter notebook containing the complete analysis, data preprocessing, visualization, and clustering steps.
- `Mall_Customers.csv`: The dataset used for analysis.

## Steps Performed

1. **Data Loading & Exploration**
   - Importing the dataset and exploring its structure.
   - Checking for missing values and basic statistics.

2. **Data Visualization**
   - Visualizing distributions of features (Age, Income, Spending Score).
   - Exploring relationships between features using scatter plots and pair plots.

3. **Data Preprocessing**
   - Encoding categorical variables (e.g., Gender).
   - Feature scaling if required.

4. **Finding the Optimal Number of Clusters**
   - Using the Elbow Method to determine the best value for K.
   - Plotting Within-Cluster-Sum-of-Squares (WCSS) for different K values.

5. **K-Means Clustering**
   - Applying the K-Means algorithm to segment customers.
   - Assigning cluster labels to each customer.

6. **Cluster Visualization & Interpretation**
   - Visualizing clusters in 2D/3D space.
   - Interpreting the characteristics of each segment.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/deathvadeR-afk/Customer_segmentation_KMeans.git
   ```
2. Open `customer_segmentation.ipynb` in Jupyter Notebook.
3. Run the cells sequentially to reproduce the analysis and results.

## Results

- Customers are segmented into distinct groups based on their spending patterns and demographics.
- Visualizations help understand the nature of each segment, which can be used for targeted marketing.

## License

This project is licensed under the MIT License.

## Author

deathvadeR-afk
