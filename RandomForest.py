#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[3]:


file_path = "C:\\Users\\Hiwi\\Downloads\\Out_of_school_rate_2022_formatted.xlsx" 
sheets_dict = pd.read_excel(file_path, sheet_name=['Primary', 'Lower secondary', 'Upper secondary'], header=[0, 1])


# In[4]:


primary_df = sheets_dict['Primary']
print(primary_df.columns.tolist()) 


# In[5]:


# Check sheet names and data
for sheet_name, df in sheets_dict.items():
    print(f"Sheet: {sheet_name}")
    print(df.head(), "\n")


# In[7]:


primary_df = sheets_dict['Primary']
print(primary_df[('Gender', 'Female')].apply(type).value_counts())#when we see our dataset the columns have different datatypes of float, int, and string data types


# In[8]:


# List of columns to clean (adjust based on our headers)
numeric_columns = [
    # Gender
    ('Gender', 'Female'), 
    ('Gender', 'Male'),
    # Residence
    ('Residence', 'Rural'), 
    ('Residence', 'Urban'),
    # Wealth quintile
    ('Wealth quintile', 'Poorest'),
    ('Wealth quintile', 'Second'),
    ('Wealth quintile', 'Middle'),
    ('Wealth quintile', 'Fourth'),
    ('Wealth quintile', 'Richest')
]

for sheet_name, df in sheets_dict.items():
    # we Converted all numeric columns to float (strings → NaN if invalid)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where critical columns (e.g., Residence_Rural) are NaN
    sheets_dict[sheet_name] = df.dropna(subset=[
        ('Gender', 'Female'),       # we tried to make sure that target column has no NaN
        ('Residence', 'Rural'),     # we tried to make sure that sure key feature has no NaN
        ('Wealth quintile', 'Poorest')  # we tried to make sure that sure another key feature has no NaN
    ])


# In[9]:


primary_df = sheets_dict['Primary']
print("Data Types After Cleaning:")
print(primary_df[numeric_columns].dtypes)

# Check for remaining NaN values
print("\nMissing Values After Cleaning:")
print(primary_df[numeric_columns].isnull().sum())


# In[10]:


# Calculate a risk score (example weights)
for sheet_name, df in sheets_dict.items():
    df[('Risk', 'Score')] = (
        0.6 * df[('Gender', 'Female')] + 
        0.3 * df[('Residence', 'Rural')] + 
        0.1 * df[('Wealth quintile', 'Poorest')]
    )
    # Categorize risk levels
    df[('Risk', 'Level')] = pd.cut(df[('Risk', 'Score')], bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'])


# In[12]:


from sklearn.model_selection import train_test_split
# Example: Using Primary school data
primary_df = sheets_dict['Primary']

# Features (X) and Target (y)
X = primary_df[[
    ('Gender', 'Female'), 
    ('Residence', 'Rural'), 
    ('Wealth quintile', 'Poorest')
]]
y = (primary_df[('Gender', 'Female')] > 0.5).astype(int)  # Binary target (1 if dropout risk > 50%)

# we Splitted data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Train the Model
from sklearn.ensemble import RandomForestClassifier
# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[13]:


# Example new data 
new_student = pd.DataFrame({
    ('Gender', 'Female'): [0.7],  
    ('Residence', 'Rural'): [1.0],  
    ('Wealth quintile', 'Poorest'): [0.8]  
})

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)[0][1]  # Probability of dropout
print(f"Predicted Dropout Risk: {probability:.0%}")


# In[14]:


#Gender Disparities: Compare mean dropout rates for girls vs. boys.

print("Female Dropout Mean:", df[('Gender', 'Female')].mean())
print("Male Dropout Mean:", df[('Gender', 'Male')].mean())


# In[17]:


#Wealth/Residence Impact: Visualize how poverty and location affect dropout rates.
sns.boxplot(x=('Wealth quintile', 'Poorest'), y=('Gender', 'Female'), data=df)
plt.title("Female Dropout Rates by Wealth Quintile")
plt.show()


# In[18]:


#Correlation Heatmap: Identify strong predictors.
sns.heatmap(df[numeric_columns].corr(), annot=True)


# In[25]:


from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Normalize all numeric columns
for sheet_name, df in sheets_dict.items():
    for col in numeric_columns:
        # Reshape to 2D array (required by sklearn)
        values = df[col].values.reshape(-1, 1)
        df[col] = scaler.fit_transform(values)
    
    # Update the sheet in the dictionary
    sheets_dict[sheet_name] = df


# In[26]:


primary_df = sheets_dict['Primary']
print("Normalized Female Dropout Rates (Sample):")
print(primary_df[('Gender', 'Female')].head())

print("\nDescriptive Statistics After Normalization:")
print(primary_df[('Gender', 'Female')].describe())


# In[19]:


#Normalize each feature to ensure equal weighting
# Normalize Female Dropout Rate (0 to 1)
df[('Normalized', 'Female_Dropout')] = df[('Gender', 'Female')] / 100

# Normalize Rural Residence (Binary: 1=Rural, 0=Urban)
df[('Normalized', 'Rural')] = df[('Residence', 'Rural')] / 100

# Normalize Poverty (Poorest=1, Richest=0)
wealth_weights = {'Poorest': 1.0, 'Second': 0.75, 'Middle': 0.5, 'Fourth': 0.25, 'Richest': 0.0}
df[('Normalized', 'Poverty')] = df[('Wealth quintile', 'Poorest')].map(wealth_weights)


# In[27]:


for sheet_name, df in sheets_dict.items():
    df[('Risk', 'Score')] = (
        0.6 * df[('Gender', 'Female')] +  # Weighted sum
        0.3 * df[('Residence', 'Rural')] +
        0.1 * df[('Wealth quintile', 'Poorest')]
    )
    sheets_dict[sheet_name] = df


# In[20]:


# Weights (sum to 1.0)
weight_female = 0.6   # Most important
weight_rural = 0.3    # Moderately important
weight_poverty = 0.1  # Least important (already partially captured in Residence)

df[('Risk', 'Score')] = (
    weight_female * df[('Normalized', 'Female_Dropout')] + 
    weight_rural * df[('Normalized', 'Rural')] + 
    weight_poverty * df[('Normalized', 'Poverty')]
)
df[('Risk', 'Score')] = df[('Risk', 'Score')] * 100


# In[21]:


bins = [0, 30, 70, 100]
labels = ['Low', 'Medium', 'High']
df[('Risk', 'Level')] = pd.cut(df[('Risk', 'Score')], bins=bins, labels=labels)


# In[28]:


wealth_labels = {
    'Poorest': 'Poorest',
    'Second': 'Second',
    'Middle': 'Middle',
    'Fourth': 'Fourth',
    'Richest': 'Richest'
}
df[('Wealth quintile', 'Label')] = df[('Wealth quintile', 'Poorest')].map(wealth_labels)

# Plot corrected boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x=('Wealth quintile', 'Label'),
    y=('Risk', 'Score'),
    data=df,
    order=['Poorest', 'Second', 'Middle', 'Fourth', 'Richest'],  # Ensure logical order
    palette='Blues'
)
plt.title("Risk Scores by Wealth Quintile", fontsize=14)
plt.xlabel("Wealth Quintile", fontsize=12)
plt.ylabel("Risk Score (0-100)", fontsize=12)
plt.ylim(0, 100)  # Force y-axis to 0-100
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.2)
plt.show()


# In[ ]:




