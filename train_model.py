# ---------------------------------------------------
# E-Commerce Sales Analysis & Revenue Prediction Model
# ---------------------------------------------------
# This script performs complete EDA and builds a 
# revenue prediction model, saving it as a .pkl file.
# Run this script to generate the model artifacts.
# ---------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

print("=" * 60)
print("   E-COMMERCE SALES ANALYSIS & REVENUE PREDICTION")
print("=" * 60)

# =============================================
# 1. DATA LOADING
# =============================================
print("\n📂 Loading dataset...")
df = pd.read_csv("ecommerce_sales_analytics_5000.csv")

print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# =============================================
# 2. DATA OVERVIEW
# =============================================
print("\n" + "=" * 60)
print("   DATA OVERVIEW")
print("=" * 60)

print("\n📋 First 5 Rows:")
print(df.head())

print("\n📋 Data Types:")
print(df.dtypes)

print("\n📋 Statistical Summary:")
print(df.describe().round(2))

print("\n📋 Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   No missing values found! ✅")

print("\n📋 Unique Values per Column:")
for col in df.columns:
    print(f"   {col}: {df[col].nunique()}")

# =============================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# =============================================
print("\n" + "=" * 60)
print("   DATA CLEANING & FEATURE ENGINEERING")
print("=" * 60)

# Convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed')

# Extract time features
df['order_year'] = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month
df['order_day_of_week'] = df['order_date'].dt.dayofweek
df['order_quarter'] = df['order_date'].dt.quarter

# Create derived features
df['total_before_discount'] = df['quantity'] * df['unit_price']
df['discount_amount'] = df['total_before_discount'] * df['discount']
df['price_per_unit_after_discount'] = df['unit_price'] * (1 - df['discount'])

print("✅ Created datetime features: year, month, day_of_week, quarter")
print("✅ Created derived features: total_before_discount, discount_amount, price_per_unit_after_discount")

# =============================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================
print("\n" + "=" * 60)
print("   EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 4.1 Revenue Distribution ---
print("\n📊 Revenue Statistics:")
print(f"   Mean Revenue:   ${df['revenue'].mean():,.2f}")
print(f"   Median Revenue: ${df['revenue'].median():,.2f}")
print(f"   Std Dev:        ${df['revenue'].std():,.2f}")
print(f"   Min Revenue:    ${df['revenue'].min():,.2f}")
print(f"   Max Revenue:    ${df['revenue'].max():,.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('E-Commerce Sales - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Revenue distribution
sns.histplot(df['revenue'], bins=50, kde=True, color='#6366f1', ax=axes[0, 0])
axes[0, 0].set_title('Revenue Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Revenue ($)')

# Revenue by category
category_revenue = df.groupby('product_category')['revenue'].mean().sort_values(ascending=True)
category_revenue.plot(kind='barh', color=['#818cf8', '#6366f1', '#4f46e5', '#4338ca'], ax=axes[0, 1])
axes[0, 1].set_title('Avg Revenue by Category', fontweight='bold')
axes[0, 1].set_xlabel('Average Revenue ($)')

# Revenue by region
region_revenue = df.groupby('region')['revenue'].mean().sort_values(ascending=True)
region_revenue.plot(kind='barh', color=['#34d399', '#10b981', '#059669', '#047857'], ax=axes[0, 2])
axes[0, 2].set_title('Avg Revenue by Region', fontweight='bold')
axes[0, 2].set_xlabel('Average Revenue ($)')

# Discount vs Revenue
axes[1, 0].scatter(df['discount'], df['revenue'], alpha=0.3, color='#f43f5e', s=10)
axes[1, 0].set_title('Discount vs Revenue', fontweight='bold')
axes[1, 0].set_xlabel('Discount (%)')
axes[1, 0].set_ylabel('Revenue ($)')

# Quantity vs Revenue
axes[1, 1].scatter(df['quantity'], df['revenue'], alpha=0.3, color='#f59e0b', s=10)
axes[1, 1].set_title('Quantity vs Revenue', fontweight='bold')
axes[1, 1].set_xlabel('Quantity')
axes[1, 1].set_ylabel('Revenue ($)')

# Payment Method distribution
payment_counts = df['payment_method'].value_counts()
axes[1, 2].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%',
               colors=['#6366f1', '#f43f5e', '#f59e0b'], startangle=90)
axes[1, 2].set_title('Payment Method Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ EDA plots saved to 'eda_plots.png'")

# --- 4.2 Correlation Analysis ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove id columns from correlation
corr_cols = [c for c in numeric_cols if c not in ['order_id', 'customer_id']]
correlation_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Correlation heatmap saved to 'correlation_heatmap.png'")

# Print top correlations with revenue
print("\n📊 Top Correlations with Revenue:")
revenue_corr = correlation_matrix['revenue'].drop('revenue').abs().sort_values(ascending=False)
for feat, corr_val in revenue_corr.head(5).items():
    direction = "+" if correlation_matrix.loc[feat, 'revenue'] > 0 else "-"
    print(f"   {direction} {feat}: {correlation_matrix.loc[feat, 'revenue']:.3f}")

# --- 4.3 Category & Region Analysis ---
print("\n📊 Sales by Product Category:")
cat_stats = df.groupby('product_category').agg(
    total_orders=('order_id', 'count'),
    total_revenue=('revenue', 'sum'),
    avg_revenue=('revenue', 'mean'),
    avg_rating=('customer_rating', 'mean')
).round(2)
print(cat_stats)

print("\n📊 Sales by Region:")
region_stats = df.groupby('region').agg(
    total_orders=('order_id', 'count'),
    total_revenue=('revenue', 'sum'),
    avg_revenue=('revenue', 'mean'),
    avg_discount=('discount', 'mean')
).round(2)
print(region_stats)

# =============================================
# 5. MODEL BUILDING
# =============================================
print("\n" + "=" * 60)
print("   MODEL BUILDING - REVENUE PREDICTION")
print("=" * 60)

# Select features for modeling
feature_cols = ['quantity', 'unit_price', 'discount', 'delivery_days',
                'customer_rating', 'order_month', 'order_quarter', 'order_day_of_week']

# Encode categorical variables
label_encoders = {}
categorical_cols = ['product_category', 'region', 'payment_method']

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    feature_cols.append(f'{col}_encoded')

print(f"\n📋 Features used ({len(feature_cols)}):")
for f in feature_cols:
    print(f"   • {f}")

X = df[feature_cols]
y = df['revenue']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📋 Data Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set:  {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5.1 Train Multiple Models ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, 
                                            min_samples_split=5, random_state=42,
                                            n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                     learning_rate=0.1, random_state=42)
}

results = {}
print("\n🔄 Training models...\n")

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Use scaled data for Linear Regression, original for tree-based
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    if name == 'Linear Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'model': model,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"   ✅ {name}:")
    print(f"      MAE:  ${mae:,.2f}")
    print(f"      RMSE: ${rmse:,.2f}")
    print(f"      R²:   {r2:.4f}")
    print(f"      CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

# --- 5.2 Model Comparison ---
print("\n" + "=" * 60)
print("   MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    name: {'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R²': r['R2'], 
           'CV R² (mean)': r['CV_Mean']}
    for name, r in results.items()
}).T.round(4)
print(f"\n{comparison_df}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_result = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} (R² = {best_result['R2']:.4f})")

# --- 5.3 Visualize Model Performance ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

colors = ['#6366f1', '#f43f5e', '#10b981']

for idx, (name, result) in enumerate(results.items()):
    axes[idx].scatter(y_test, result['predictions'], alpha=0.4, color=colors[idx], s=15)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'k--', linewidth=1.5, label='Perfect Prediction')
    axes[idx].set_title(f'{name}\nR² = {result["R2"]:.4f}', fontweight='bold')
    axes[idx].set_xlabel('Actual Revenue ($)')
    axes[idx].set_ylabel('Predicted Revenue ($)')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Model comparison plot saved to 'model_comparison.png'")

# --- 5.4 Feature Importance (Best Model) ---
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_model = best_result['model']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['feature'], feature_importance['importance'], color='#6366f1')
    ax.set_title(f'Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance plot saved to 'feature_importance.png'")
    
    print(f"\n📊 Top 5 Most Important Features:")
    for _, row in feature_importance.tail(5).iloc[::-1].iterrows():
        print(f"   • {row['feature']}: {row['importance']:.4f}")

# =============================================
# 6. SAVE MODEL & ARTIFACTS
# =============================================
print("\n" + "=" * 60)
print("   SAVING MODEL & ARTIFACTS")
print("=" * 60)

# Save the best model
best_model_obj = best_result['model']
joblib.dump(best_model_obj, 'revenue_model.pkl')
print(f"\n✅ Best model ({best_model_name}) saved to 'revenue_model.pkl'")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved to 'scaler.pkl'")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Label encoders saved to 'label_encoders.pkl'")

# Save feature columns list
joblib.dump(feature_cols, 'feature_columns.pkl')
print("✅ Feature columns saved to 'feature_columns.pkl'")

# Save model metadata
model_metadata = {
    'best_model_name': best_model_name,
    'features': feature_cols,
    'categorical_columns': categorical_cols,
    'metrics': {
        'MAE': best_result['MAE'],
        'RMSE': best_result['RMSE'],
        'R2': best_result['R2'],
        'CV_R2_mean': best_result['CV_Mean'],
        'CV_R2_std': best_result['CV_Std']
    },
    'all_model_results': {
        name: {'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2']}
        for name, r in results.items()
    },
    'category_values': df['product_category'].unique().tolist(),
    'region_values': df['region'].unique().tolist(),
    'payment_values': df['payment_method'].unique().tolist(),
    'data_stats': {
        'total_records': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'revenue_mean': float(df['revenue'].mean()),
        'revenue_std': float(df['revenue'].std()),
    }
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print("✅ Model metadata saved to 'model_metadata.pkl'")

# Save the processed dataset for Streamlit
df.to_csv('processed_data.csv', index=False)
print("✅ Processed dataset saved to 'processed_data.csv'")

print("\n" + "=" * 60)
print("   ALL DONE! 🎉")
print("=" * 60)
print(f"\n   📁 Files created:")
print(f"      • revenue_model.pkl     (trained {best_model_name})")
print(f"      • scaler.pkl            (feature scaler)")
print(f"      • label_encoders.pkl    (categorical encoders)")
print(f"      • feature_columns.pkl   (feature list)")
print(f"      • model_metadata.pkl    (metrics & metadata)")
print(f"      • processed_data.csv    (processed dataset)")
print(f"      • eda_plots.png         (EDA visualizations)")
print(f"      • correlation_heatmap.png")
print(f"      • model_comparison.png")
print(f"      • feature_importance.png")
print(f"\n   🚀 Run the Streamlit app:")
print(f"      streamlit run app.py")
