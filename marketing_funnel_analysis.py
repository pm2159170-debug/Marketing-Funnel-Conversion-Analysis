"""
Marketing Funnel & Conversion Performance Analysis
Future Interns - Data Science & Analytics Task 3 (2026)
Analyst: Prasad Manik
Tool: Python
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("="*60)
print("STEP 1 - Loading Data")
print("="*60)
df = pd.read_csv('marketing_funnel_data.csv', parse_dates=['date'])
print(f"Dataset shape: {df.shape}")
print(df.head())

print("\n" + "="*60)
print("STEP 2 - Data Overview")
print("="*60)
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

print("\n" + "="*60)
print("STEP 3 - Funnel Stage Conversion Rates")
print("="*60)
funnel = {
    'Visitors': df['visitors'].sum(),
    'Leads': df['leads'].sum(),
    'MQLs': df['mqls'].sum(),
    'SQLs': df['sqls'].sum(),
    'Opportunities': df['opportunities'].sum(),
    'Customers': df['customers'].sum()
}
funnel_df = pd.DataFrame(list(funnel.items()), columns=['Stage', 'Count'])
funnel_df['Conversion_from_Previous'] = funnel_df['Count'] / funnel_df['Count'].shift(1)
funnel_df['Conversion_from_Visitors'] = funnel_df['Count'] / funnel_df.loc[0, 'Count']
funnel_df = funnel_df.fillna(1.0)

display_df = funnel_df.copy()
display_df['Conversion_from_Previous'] = display_df['Conversion_from_Previous'].map('{:.1%}'.format)
display_df['Conversion_from_Visitors'] = display_df['Conversion_from_Visitors'].map('{:.1%}'.format)
print(display_df.to_string(index=False))

print("\n" + "="*60)
print("STEP 4 - Generating Funnel Chart")
print("="*60)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = ['#1F3864','#2E75B6','#3498DB','#5DADE2','#85C1E9','#AED6F1']
bars = axes[0].barh(funnel_df['Stage'][::-1], funnel_df['Count'][::-1], color=colors, height=0.6)
axes[0].set_title('Marketing Funnel - Volume at Each Stage', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Total Count', fontsize=11)
for bar, val in zip(bars, funnel_df['Count'][::-1]):
    axes[0].text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=10, fontweight='bold')
axes[0].set_xlim(0, funnel_df['Count'].max() * 1.15)

stages = funnel_df['Stage'][1:].tolist()
dropoffs = (1 - funnel_df['Conversion_from_Previous'][1:]).tolist()
drop_colors = ['#E74C3C' if d > 0.5 else '#E67E22' if d > 0.3 else '#F1C40F' for d in dropoffs]
bars2 = axes[1].bar(stages, [d*100 for d in dropoffs], color=drop_colors, width=0.5)
axes[1].set_title('Drop-off Rate at Each Funnel Stage (%)', fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Drop-off Rate (%)', fontsize=11)
axes[1].set_ylim(0, 100)
for bar, val in zip(bars2, dropoffs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
axes[1].axhline(50, color='red', linestyle='--', alpha=0.4, label='50% threshold')
axes[1].legend(fontsize=9)
plt.tight_layout(pad=3)
plt.savefig('funnel_overview.png', dpi=150, bbox_inches='tight')
print("Saved: funnel_overview.png")
plt.close()

print("\n" + "="*60)
print("STEP 5 - Channel Performance Analysis")
print("="*60)
channel = df.groupby('channel').agg(
    Visitors=('visitors','sum'),
    Leads=('leads','sum'),
    MQLs=('mqls','sum'),
    SQLs=('sqls','sum'),
    Customers=('customers','sum'),
    Total_Cost=('lead_gen_cost','sum'),
    Total_Revenue=('revenue','sum')
).reset_index()

channel['Visitor_to_Lead'] = (channel['Leads'] / channel['Visitors'] * 100).round(1)
channel['Lead_to_Customer'] = (channel['Customers'] / channel['Leads'] * 100).round(1)
channel['Overall_Conversion'] = (channel['Customers'] / channel['Visitors'] * 100).round(2)
channel['ROI'] = ((channel['Total_Revenue'] - channel['Total_Cost']) / channel['Total_Cost'] * 100).round(1)
channel['Cost_per_Customer'] = (channel['Total_Cost'] / channel['Customers']).round(2)
channel = channel.sort_values('Overall_Conversion', ascending=False)

print(channel[['channel','Visitors','Leads','Customers','Visitor_to_Lead',
               'Lead_to_Customer','Overall_Conversion','ROI','Cost_per_Customer']].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ch_colors = ['#1F3864','#2E75B6','#E67E22','#1E8449','#8E44AD','#C0392B']

sorted_ch = channel.sort_values('Overall_Conversion')
axes[0].barh(sorted_ch['channel'], sorted_ch['Overall_Conversion'], color=ch_colors, height=0.5)
axes[0].set_title('Overall Conversion Rate\nby Channel (%)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Visitor to Customer (%)')
for i, val in enumerate(sorted_ch['Overall_Conversion']):
    axes[0].text(val + 0.02, i, f'{val}%', va='center', fontsize=9, fontweight='bold')

sorted_roi = channel.sort_values('ROI')
roi_colors = ['#E74C3C' if r < 0 else '#1E8449' for r in sorted_roi['ROI']]
axes[1].barh(sorted_roi['channel'], sorted_roi['ROI'], color=roi_colors, height=0.5)
axes[1].set_title('ROI by Channel (%)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('ROI (%)')
axes[1].axvline(0, color='black', linewidth=0.8)
for i, val in enumerate(sorted_roi['ROI']):
    axes[1].text(val + 20, i, f'{val}%', va='center', fontsize=9, fontweight='bold')

sorted_cpc = channel.sort_values('Cost_per_Customer', ascending=False)
axes[2].barh(sorted_cpc['channel'], sorted_cpc['Cost_per_Customer'], color=ch_colors, height=0.5)
axes[2].set_title('Cost per Customer\nby Channel (Rs)', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Cost per Customer (Rs)')
for i, val in enumerate(sorted_cpc['Cost_per_Customer']):
    axes[2].text(val + 1, i, f'Rs{val}', va='center', fontsize=9, fontweight='bold')

plt.suptitle('Channel Performance Analysis', fontsize=15, fontweight='bold')
plt.tight_layout(pad=3)
plt.savefig('channel_performance.png', dpi=150, bbox_inches='tight')
print("Saved: channel_performance.png")
plt.close()

print("\n" + "="*60)
print("STEP 6 - Monthly Conversion Trend")
print("="*60)
df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').agg(
    Visitors=('visitors','sum'),
    Leads=('leads','sum'),
    Customers=('customers','sum'),
    Revenue=('revenue','sum')
).reset_index()
monthly['month_str'] = monthly['month'].astype(str)
monthly['Conv_Rate'] = (monthly['Customers'] / monthly['Visitors'] * 100).round(2)
monthly['Lead_Rate'] = (monthly['Leads'] / monthly['Visitors'] * 100).round(1)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].fill_between(monthly['month_str'], monthly['Revenue'], alpha=0.3, color='#2E75B6')
axes[0].plot(monthly['month_str'], monthly['Revenue'], 'o-', color='#1F3864', linewidth=2.5, markersize=7)
axes[0].set_title('Monthly Revenue Trend', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Revenue (Rs)')
for x, y in zip(monthly['month_str'], monthly['Revenue']):
    axes[0].text(x, y + 500, f'Rs{y:,}', ha='center', fontsize=9, fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

axes[1].plot(monthly['month_str'], monthly['Conv_Rate'], 's-', color='#E67E22',
             linewidth=2.5, markersize=8, label='Visitor to Customer %')
axes[1].plot(monthly['month_str'], monthly['Lead_Rate'], '^-', color='#1E8449',
             linewidth=2.5, markersize=8, label='Visitor to Lead %')
axes[1].set_title('Monthly Conversion Rate Trend', fontweight='bold', fontsize=13)
axes[1].set_ylabel('Conversion Rate (%)')
axes[1].legend(fontsize=10)
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Monthly Performance Trends', fontsize=15, fontweight='bold')
plt.tight_layout(pad=3)
plt.savefig('monthly_trends.png', dpi=150, bbox_inches='tight')
print("Saved: monthly_trends.png")
plt.close()

print("\n" + "="*60)
print("STEP 7 - Key Insights & Recommendations")
print("="*60)

total_visitors = df['visitors'].sum()
total_customers = df['customers'].sum()
total_revenue = df['revenue'].sum()
total_cost = df['lead_gen_cost'].sum()
overall_conv = total_customers / total_visitors * 100
overall_roi = (total_revenue - total_cost) / total_cost * 100

best_channel = channel.loc[channel['Overall_Conversion'].idxmax(), 'channel']
worst_channel = channel.loc[channel['Overall_Conversion'].idxmin(), 'channel']
best_roi = channel.loc[channel['ROI'].idxmax(), 'channel']

print("="*60)
print("       MARKETING FUNNEL - EXECUTIVE SUMMARY")
print("="*60)
print(f"  Total Visitors:        {total_visitors:,}")
print(f"  Total Customers:       {total_customers:,}")
print(f"  Overall Conv. Rate:    {overall_conv:.2f}%")
print(f"  Total Revenue:         Rs{total_revenue:,}")
print(f"  Total Marketing Cost:  Rs{total_cost:,}")
print(f"  Overall ROI:           {overall_roi:.1f}%")
print(f"  Best Channel:          {best_channel}")
print(f"  Worst Channel:         {worst_channel}")
print(f"  Highest ROI Channel:   {best_roi}")
print("="*60)
print()
print("KEY FINDINGS:")
print("  1. Visitors to Leads: Biggest drop-off (~84% lost) - the single largest leak in the funnel")
print("  2. Referral has the highest overall conversion rate (4.03%)")
print("  3. Email has the highest ROI (3515%) - extremely cost-efficient channel")
print("  4. Social Media has the lowest overall conversion (0.64%) and lowest ROI")
print("  5. Paid Search and Social Media have very high cost per customer (Rs102 and Rs180)")
print()
print("RECOMMENDATIONS:")
print("  1. Scale up Email and Referral campaigns - best ROI and conversion combination")
print("  2. Re-evaluate Social Media strategy - high spend, lowest returns")
print("  3. Improve Visitor-to-Lead conversion through better landing pages and CTAs")
print("  4. Reduce Paid Search budget or renegotiate cost-per-click rates")
print("  5. Use Referral incentive programs to replicate Email's high ROI at scale")
print("="*60)

print("\nALL CHARTS SAVED SUCCESSFULLY!")
print("Files created: funnel_overview.png, channel_performance.png, monthly_trends.png")
