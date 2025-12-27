import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate dates for the past 2 years
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start_date, end_date, freq='D')

# Sample data
products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch', 
            'Camera', 'Mouse', 'Keyboard', 'Monitor', 'Speaker']
categories = ['Electronics', 'Electronics', 'Electronics', 'Audio', 'Electronics',
              'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Audio']
regions = ['North', 'South', 'East', 'West']
customer_segments = ['Consumer', 'Corporate', 'Small Business']

# Generate 5000 sales records
n_records = 5000
data = []

for i in range(n_records):
    product_idx = random.randint(0, len(products) - 1)
    date = random.choice(dates)
    
    # Price based on product
    base_prices = {'Laptop': 800, 'Smartphone': 600, 'Tablet': 400, 
                   'Headphones': 150, 'Smartwatch': 250, 'Camera': 500,
                   'Mouse': 30, 'Keyboard': 80, 'Monitor': 300, 'Speaker': 120}
    
    base_price = base_prices[products[product_idx]]
    quantity = random.randint(1, 5)
    discount = random.choice([0, 0.05, 0.1, 0.15, 0.2])
    price = base_price * (1 - discount) * quantity
    
    # Add some seasonal variation
    if date.month in [11, 12]:  # Holiday season
        price *= 1.2
    elif date.month in [6, 7]:  # Summer sales
        price *= 0.9
    
    data.append({
        'OrderID': f'ORD{i+1000:06d}',
        'Date': date,
        'Product': products[product_idx],
        'Category': categories[product_idx],
        'Region': random.choice(regions),
        'CustomerSegment': random.choice(customer_segments),
        'Quantity': quantity,
        'UnitPrice': base_price * (1 - discount),
        'TotalPrice': price,
        'Discount': discount * 100,
        'ShippingCost': random.choice([0, 5, 10, 15])
    })

df = pd.DataFrame(data)
df['Profit'] = df['TotalPrice'] * 0.25  # Assume 25% profit margin
df['NetRevenue'] = df['TotalPrice'] - df['ShippingCost']

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print(f"Generated {len(df)} sales records")
print(df.head())
print(f"\nData saved to sales_data.csv")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading data...")
df = pd.read_csv('sales_data.csv', parse_dates=['Date'])
print(f"Data loaded: {len(df)} records")

# Data Cleaning
print("\n=== DATA CLEANING ===")
print(f"Initial shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Remove any duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Add derived columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['MonthName'] = df['Date'].dt.strftime('%B')
df['Quarter'] = df['Date'].dt.quarter
df['DayOfWeek'] = df['Date'].dt.day_name()
df['YearMonth'] = df['Date'].dt.to_period('M')

print("\n=== DATA OVERVIEW ===")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nSummary statistics:\n{df.describe()}")

# ==================== ANALYSIS ====================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# 1. Overall Sales Metrics
print("\n1. OVERALL SALES METRICS")
total_revenue = df['NetRevenue'].sum()
total_profit = df['Profit'].sum()
total_orders = len(df)
avg_order_value = df['NetRevenue'].mean()
profit_margin = (total_profit / total_revenue) * 100

print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Average Order Value: ${avg_order_value:,.2f}")
print(f"Profit Margin: {profit_margin:.2f}%")

# 2. Sales by Product
print("\n2. TOP 10 PRODUCTS BY REVENUE")
product_sales = df.groupby('Product').agg({
    'NetRevenue': 'sum',
    'Quantity': 'sum',
    'OrderID': 'count'
}).sort_values('NetRevenue', ascending=False)
product_sales.columns = ['TotalRevenue', 'TotalQuantity', 'OrderCount']
print(product_sales.head(10))

# 3. Sales by Category
print("\n3. SALES BY CATEGORY")
category_sales = df.groupby('Category').agg({
    'NetRevenue': 'sum',
    'Profit': 'sum',
    'OrderID': 'count'
}).sort_values('NetRevenue', ascending=False)
category_sales.columns = ['TotalRevenue', 'TotalProfit', 'OrderCount']
print(category_sales)

# 4. Sales by Region
print("\n4. SALES BY REGION")
region_sales = df.groupby('Region').agg({
    'NetRevenue': 'sum',
    'OrderID': 'count'
}).sort_values('NetRevenue', ascending=False)
region_sales.columns = ['TotalRevenue', 'OrderCount']
print(region_sales)

# 5. Sales by Customer Segment
print("\n5. SALES BY CUSTOMER SEGMENT")
segment_sales = df.groupby('CustomerSegment').agg({
    'NetRevenue': 'sum',
    'OrderID': 'count',
    'NetRevenue': 'mean'
}).sort_values('NetRevenue', ascending=False)
print(segment_sales)

# 6. Monthly Sales Trend
print("\n6. MONTHLY SALES TREND")
monthly_sales = df.groupby('YearMonth').agg({
    'NetRevenue': 'sum',
    'OrderID': 'count'
}).reset_index()
monthly_sales.columns = ['Month', 'Revenue', 'Orders']
print(monthly_sales.tail(12))

# ==================== VISUALIZATIONS ====================

print("\n" + "="*50)
print("GENERATING VISUALIZATIONS...")
print("="*50)

# Create output directory for plots
os.makedirs('visualizations', exist_ok=True)

# 1. Monthly Sales Trend
plt.figure(figsize=(14, 6))
monthly_revenue = df.groupby('YearMonth')['NetRevenue'].sum()
monthly_revenue.plot(kind='line', marker='o', linewidth=2, markersize=6)
plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/monthly_sales_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: monthly_sales_trend.png")

# 2. Top 10 Products by Revenue
plt.figure(figsize=(12, 6))
top_products = df.groupby('Product')['NetRevenue'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='barh', color='steelblue')
plt.title('Top 10 Products by Revenue', fontsize=16, fontweight='bold')
plt.xlabel('Revenue ($)', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/top_products.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: top_products.png")

# 3. Sales by Category
plt.figure(figsize=(10, 6))
category_revenue = df.groupby('Category')['NetRevenue'].sum().sort_values(ascending=False)
colors = sns.color_palette("husl", len(category_revenue))
category_revenue.plot(kind='bar', color=colors)
plt.title('Sales by Category', fontsize=16, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/sales_by_category.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: sales_by_category.png")

# 4. Sales by Region
plt.figure(figsize=(10, 6))
region_revenue = df.groupby('Region')['NetRevenue'].sum().sort_values(ascending=False)
region_revenue.plot(kind='bar', color='coral')
plt.title('Sales by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/sales_by_region.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: sales_by_region.png")

# 5. Customer Segment Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

segment_revenue = df.groupby('CustomerSegment')['NetRevenue'].sum()
segment_revenue.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
ax1.set_title('Revenue Distribution by Customer Segment', fontsize=14, fontweight='bold')
ax1.set_ylabel('')

segment_avg_order = df.groupby('CustomerSegment')['NetRevenue'].mean()
segment_avg_order.plot(kind='bar', ax=ax2, color='teal')
ax2.set_title('Average Order Value by Segment', fontsize=14, fontweight='bold')
ax2.set_xlabel('Customer Segment', fontsize=12)
ax2.set_ylabel('Average Order Value ($)', fontsize=12)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/customer_segment_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: customer_segment_analysis.png")

# 6. Quarterly Sales Comparison
plt.figure(figsize=(12, 6))
quarterly_sales = df.groupby(['Year', 'Quarter'])['NetRevenue'].sum().reset_index()
quarterly_sales['QuarterLabel'] = quarterly_sales['Year'].astype(str) + ' Q' + quarterly_sales['Quarter'].astype(str)

sns.barplot(data=quarterly_sales, x='QuarterLabel', y='NetRevenue', palette='viridis')
plt.title('Quarterly Sales Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/quarterly_sales.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: quarterly_sales.png")

# 7. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'Discount', 'ShippingCost', 'Profit', 'NetRevenue']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# 8. Sales Distribution by Day of Week
plt.figure(figsize=(10, 6))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales = df.groupby('DayOfWeek')['NetRevenue'].sum().reindex(day_order)
day_sales.plot(kind='bar', color='mediumpurple')
plt.title('Sales Distribution by Day of Week', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/day_of_week_sales.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: day_of_week_sales.png")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print(f"\nAll visualizations saved in 'visualizations' folder")
print(f"Total visualizations created: 8")