import pandas as pd
import numpy as np

def calculate_trend_shifts(df):
    """
    Groups data by month and trend, calculates growth and categorizes shift.
    df must have 'date' and 'detected_trends' (list)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month_year'] = df['date'].dt.to_period('M')
    
    # Explode the list of trends so each trend gets its own row
    exploded_df = df.explode('detected_trends').dropna(subset=['detected_trends'])
    
    # Group by month and trend
    monthly_counts = exploded_df.groupby(['month_year', 'detected_trends']).size().reset_index(name='mentions')
    
    # Sort by trend and month
    monthly_counts = monthly_counts.sort_values(by=['detected_trends', 'month_year'])
    
    # Calculate previous period mentions
    monthly_counts['prev_mentions'] = monthly_counts.groupby('detected_trends')['mentions'].shift(1)
    monthly_counts['prev_mentions'] = monthly_counts['prev_mentions'].fillna(0)
    
    # Calculate growth rate
    def calc_growth(row):
        if row['prev_mentions'] == 0:
            return 1.0 if row['mentions'] > 0 else 0.0
        return (row['mentions'] - row['prev_mentions']) / row['prev_mentions']
        
    monthly_counts['growth_rate'] = monthly_counts.apply(calc_growth, axis=1)
    
    # Categorize shift
    def categorize_trend(growth):
        if growth >= 0.5:
            return "Emerging"
        elif 0.1 <= growth < 0.5:
            return "Rising"
        elif -0.1 <= growth < 0.1:
            return "Stable"
        else:
            return "Declining"
            
    monthly_counts['status'] = monthly_counts['growth_rate'].apply(categorize_trend)
    
    return monthly_counts

if __name__ == "__main__":
    pass
