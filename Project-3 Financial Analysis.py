#!/usr/bin/env python
# coding: utf-8

# # Project-3 Financial Analysis

# In[1]:


# Submitted by: Umang Parti
# Submitted to: Unified Mentor


# In[2]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from IPython.display import FileLink


# In[3]:


# importing dataset
finance_df = pd.read_csv(r"C:\Users\umang\Desktop\unified mentor\project - 3\Financial_data.csv")
print(finance_df.head())


# In[4]:


# Cleaning the dataset
finance_df.isnull().sum()


# - We have missing values in 9 rows of Market Capital Column 
# - And missing values in 30 rows of Quarterly Sales

# In[5]:


# Impute missing values (replace NaN with mean, median, or other strategies)
finance_df['Mar Cap - Crore'].fillna(finance_df['Mar Cap - Crore'].mean(), inplace=True)


# In[6]:


# Impute missing values (replace NaN with mean, median, or other strategies)
finance_df['Sales Qtr - Crore'].fillna(finance_df['Sales Qtr - Crore'].mean(), inplace=True)


# # Exploratory Analysis

# In[7]:


finance_df.shape


# we can see that
# - Our data of 488 companies
# - Our data has four columns of serial number, company name, sales quarter and market capitalization

# In[8]:


finance_df.columns


# In[9]:


finance_df.info()


# In[10]:


finance_df.describe()


# - Mean Market Capital is 28 thousand crores
# - Maximum market capital is 583 thousand crore
# - Minimum market capital is 3 thousand crore
# - Mean Sales Quarter is 38 hundred crore
# - Maximum Sales Quarter is 110 thousand crore
# - Minimum Sales Quarter is 19 crore

# # Identifying Outliers

# In[11]:


# Plotting boxplot for Market Capital
plt.figure(figsize=(8, 8))
sns.boxplot(finance_df['Mar Cap - Crore'])
plt.title('Boxplot with Outliers')
plt.savefig('my_chart.png')
plt.show()


# In[12]:


FileLink(r'my_chart.png')


# In[13]:


# Function to get lower and upper critical values
def outlier_detection(df_marketcap):
    df_marketcap_cleaned = df_marketcap.dropna()
    if len(df_marketcap_cleaned) == 0:
        raise ValueError("No valid data after removing NaN values.")
    df_marketcap_sorted = sorted(df_marketcap_cleaned)
    Q1 = np.percentile(df_marketcap_sorted, 25)
    Q3 = np.percentile(df_marketcap_sorted, 75)
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


# In[14]:


lower, upper = outlier_detection(finance_df['Mar Cap - Crore'])
print(lower, upper)


# In[15]:


# identifying outliers
outliers= (finance_df['Mar Cap - Crore'] >=upper) | (finance_df['Mar Cap - Crore'] <= lower)
outliers.value_counts()


# - Out of 488 records our 57 records are considered as outliers

# In[16]:


# Function to remove outliers based on IQR
def remove_outliers_iqr(df, column_name):
    df_no_outliers = df[(df[column_name] >= lower) & (df[column_name] <= upper)]
    return df_no_outliers


# In[17]:


finance_df_no_outliers = remove_outliers_iqr(finance_df, 'Mar Cap - Crore')
print(finance_df_no_outliers)


# In[18]:


finance_df_no_outliers.info()


# In[19]:


finance_df_no_outliers_reset = finance_df_no_outliers.reset_index(drop=True)
plt.figure(figsize=(5.5,5.5))
sns.boxplot(finance_df_no_outliers_reset['Mar Cap - Crore'])
plt.title('Boxplot without outliers')
plt.savefig('my_chart2.png')
plt.show()


# In[20]:


FileLink(r'my_chart2.png')


# - We have successfully removed the Outliers. 
# - Earlier the maximum Market Capitalization was approx 6 lakhs crores 
# - Now Distribution is even and maximum market capitalization is 50 thousand crore

# # Market Capitalization Distribution

# In[21]:


# Sort the Data by market capitalization in descending order
finance_df_sorted = finance_df.sort_values(by='Mar Cap - Crore', ascending=False)
print(finance_df_sorted)


# In[22]:


# We will group the data in an interval of 50 each
bin_labels = ['500-450','450-400','400-350','350-300','300-250','250-200','200-150',
              '150-100','100-50','50-0','0']
bin_edges = np.arange(0, 551, 50)


# In[23]:


finance_df_sorted['Groups'] = pd.cut(finance_df_sorted['S.No.'], 
                                         bins=bin_edges, labels=bin_labels, right=False)
finance_df_sorted['Groups'] = pd.Categorical(finance_df_sorted['Groups'], categories=bin_labels,
                                             ordered=True)
finance_df_sorted


# In[24]:


# Creating bar chart and plotting the companies in agroup of 50 and their market capital
plt.figure(figsize=(8, 8))
plt.bar(finance_df_sorted['Groups'], finance_df_sorted['Mar Cap - Crore'], width=0.6)
plt.title('Market Capital of Companies Group Wise')
plt.xticks(finance_df_sorted['Groups'].cat.categories)
plt.savefig('my_chart3.png')
plt.show()


# In[25]:


FileLink(r'my_chart3.png')


# In[26]:


pip install -U kaleido


# In[27]:


fig = px.treemap(finance_df_sorted, path=['Name'], values='Mar Cap - Crore')
fig.update_layout(width=700,height=700)
fig.update_layout(title_text='Treemap of Companies and Market Capital')
fig.show()
fig.write_image('treemap_chart.png')


# In[28]:


FileLink(r'treemap_chart.png')


# In[29]:


finance_df_sorted.drop(columns='Groups',inplace = True)


# # Top 50 Companies

# In[30]:


# Select the top 50 companies
top_50_companies = finance_df_sorted.head(50)
print(top_50_companies.head())


# In[31]:


# To get to 50 companies and their share in market capitalization
# Calculate the total market capitalization of the top 500 companies
total_market_cap_top_500 = finance_df['Mar Cap - Crore'].sum()


# In[32]:


# Calculate the share of each company in the total market capitalization
top_50_companies['MarketCapShare'] = (top_50_companies['Mar Cap - Crore']/
                                           total_market_cap_top_500 * 100)


# In[33]:


# Print the top 50 companies
print(top_50_companies.head(20))


# In[34]:


plt.figure(figsize=(6,6))
top_50_companies['MarketCapShare'].plot()
plt.savefig('my_chart5.png')
plt.title('Market Capital top 50 companies')
plt.show()


# In[35]:


FileLink(r'my_chart5.png')


# In[36]:


print(top_50_companies['MarketCapShare'].head(6).sum())
print(top_50_companies['MarketCapShare'].head(15).sum())


# - We Notice that Top 6 companies contribute around 18.47% of Market Share
# - Top 15 companies contribute to 32.64% of Market Share

# # Top 15 Dominant Players in the Market

# In[37]:


# Select the top 15 companies
top_15_companies = finance_df_sorted.head(15)
print(top_15_companies)


# In[38]:


# creating a piechart
labels = top_15_companies['Name'] 
sizes = top_15_companies['Mar Cap - Crore']


# In[39]:


plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Top 15 Dominant Players')
plt.savefig('my_chart6.png')
plt.show()


# In[40]:


FileLink(r'my_chart6.png')


# # Market Capital and Quarterly Sales 

# In[41]:


# Scatter Plot


# In[42]:


# Plotting the scatter plot with regression line
sns.scatterplot(x='Mar Cap - Crore', y='Sales Qtr - Crore', data=finance_df_no_outliers)
# for our analysis we will use the data without outliers
# Plot the regression line
sns.regplot(x='Mar Cap - Crore', y='Sales Qtr - Crore', data=finance_df_no_outliers, 
            scatter_kws={'s': 100}, line_kws={'color': 'red'})
plt.xlabel('Market Capitalization')
plt.ylabel('Quarterly Sales')
plt.title('Scatter Plot with Regression Line without outliers')
plt.savefig('my_chart7.png')
plt.show()


# In[43]:


FileLink(r'my_chart7.png')


# - We have Market Capital on X axis and Quaterly Sales on Y axis
# - Mostly the points are clustered closed to 0 depicting that lower the market capital lower is the quaterly sales
# - With the help of regression line we are able to see an upward positive trend between the variables more carefully
# - The increase in Quarterly sales along with many other factors lead to increase in market capital

# In[44]:


# Let us plot a scatter plot with Outliers as well
# Plotting the scatter plot with regression line
sns.scatterplot(x='Mar Cap - Crore', y='Sales Qtr - Crore', data=finance_df)

# Plot the regression line
sns.regplot(x='Mar Cap - Crore', y='Sales Qtr - Crore', data=finance_df, scatter_kws={'s': 100},
            line_kws={'color': 'red'})
plt.xlabel('Market Capitalization')
plt.ylabel('Quarterly Sales')
plt.title('Scatter Plot with Regression Line')
plt.savefig('my_chart8.png')
plt.show()


# In[45]:


FileLink(r'my_chart8.png')


# - Now we see a positive upward slopping line more clearly and accurately

# In[46]:


# Correlation Analysis


# In[47]:


# To find correlation between the two variables
correlation_coefficient = finance_df['Mar Cap - Crore'].corr(finance_df['Sales Qtr - Crore'])
print("Correlation Coefficient:",correlation_coefficient)


# - We find that we have a positive correlation between the variables of 0.62 which is fairly significant

# In[48]:


# Regression Analysis


# In[49]:


X = sm.add_constant(finance_df['Mar Cap - Crore'])
X


# In[50]:


model = sm.OLS(finance_df['Sales Qtr - Crore'], X).fit()


# In[51]:


print(model.summary())


# - F-statistic of 307 typically expect a very low p-value (close to zero), indicating that the overall model is statistically significant.
# - R-squared of 0.387 means that approximately 38.7% of the variability in the dependent variable is explained by the independent variables included in your model.
# - Adjusted R-squared of 0.386 is slightly lower than the R-squared. This indicates that the additional predictors in your model are not contributing much explanatory power.
# - The coefficient for the variable "Mar Cap - Crore" is 0.1023.
# - The standard error of the coefficient is 0.006, indicating the precision of the estimate.
# - The t-statistic of 17.521 is highly significant, suggesting that the variable is strongly related to the dependent variable.
# - The p-value (0.000) is less than the significance level, indicating the statistical significance of "Mar Cap - Crore."
# - The 95% confidence interval for the coefficient is [0.091, 0.114], providing a range for the true population coefficient.

# In[52]:


#...............................................................................................

