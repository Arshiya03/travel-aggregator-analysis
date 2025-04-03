import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
try:
    bookings_df = pd.read_csv('Bookings.csv')
    sessions_df = pd.read_csv('Sessions.csv')
except FileNotFoundError:
    print("Error: One or more data files not found. Please ensure 'Bookings.csv' and 'Sessions.csv' are in the same directory.")
    exit()

# Drop null values in booking_id column
bookings_df.dropna(subset=['booking_id'], inplace=True)
sessions_df.dropna(subset=['booking_id'], inplace=True)

# 1. Number of distinct bookings, sessions, and searches
distinct_bookings_bookings_df = bookings_df['booking_id'].nunique()
distinct_bookings_sessions_df = sessions_df['booking_id'].nunique()
distinct_sessions = sessions_df['session_id'].nunique()
distinct_searches = sessions_df['search_id'].nunique()


print(f"Number of distinct bookings in bookings_df: {distinct_bookings_bookings_df}")
print(f"Number of distinct bookings in sessions_df: {distinct_bookings_sessions_df}")
print(f"Number of distinct sessions: {distinct_sessions}")
print(f"Number of distinct searches: {distinct_searches}")

# 2. How many sessions have more than one booking
# Merge bookings_df and sessions_df on booking_id
merged_df = pd.merge(bookings_df, sessions_df, on='booking_id', how='inner')

# Count bookings per session
bookings_per_session = merged_df.groupby('session_id')['booking_id'].count()

# Filter sessions with more than one booking
sessions_multiple_bookings = bookings_per_session[bookings_per_session > 1].count()

print(f"\nNumber of sessions with more than one booking: {sessions_multiple_bookings}")

# 3. Which days of the week have the highest number of bookings? Also, draw a pie chart
bookings_df['booking_time'] = pd.to_datetime(bookings_df['booking_time'])
bookings_df['day_of_week'] = bookings_df['booking_time'].dt.day_name()

weekday_counts = bookings_df['day_of_week'].value_counts().sort_values(ascending=False)
print("\nNumber of bookings per day of the week:\n", weekday_counts)
print("\nDay of the week with the highest number of bookings:", weekday_counts.index[0])

# Pie chart for booking distribution across days of the week
plt.figure(figsize=(8, 8))
plt.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Bookings Across Days of the Week')
plt.show()

# 4. For each of the service names, display the total number of bookings and the total Gross Booking Value in INR
service_summary = bookings_df.groupby('service_name').agg(
    total_bookings=('booking_id', 'count'),
    total_gbv_inr=('INR_Amount', 'sum')
)
print("\nTotal bookings and Gross Booking Value (INR) per service:\n", service_summary)

# 5. For customers who have more than 1 booking, which is the most booked route (from_city to to_city)?
customer_booking_counts = bookings_df['customer_id'].value_counts()
customers_multiple_bookings = customer_booking_counts[customer_booking_counts > 1].index
multiple_booking_df = bookings_df[bookings_df['customer_id'].isin(customers_multiple_bookings)]
most_booked_route = multiple_booking_df.groupby(['from_city', 'to_city']).size().sort_values(ascending=False).head(1)

print("\nMost booked route for customers with more than 1 booking:\n", most_booked_route)

# 6. Which are the top 3 departure cities from where customers book mostly in advance, provided that there have been at least 5 departures from that city?
bookings_df['departure_timestamp'] = bookings_df['booking_time'].copy()
bookings_df['departure_timestamp'] = pd.to_datetime(bookings_df['departure_timestamp'])
bookings_df['days_in_advance'] = bookings_df['days_to_departure'].copy()

departure_city_counts = bookings_df['from_city'].value_counts()
frequent_departure_cities = departure_city_counts[departure_city_counts >= 5].index
frequent_departures_df = bookings_df[bookings_df['from_city'].isin(frequent_departure_cities)]

advance_booking_analysis = frequent_departures_df.groupby('from_city')['days_to_departure'].mean().sort_values(ascending=False)
top_3_advance_departure_cities = advance_booking_analysis.head(3)

print("\nTop 3 departure cities with mostly advance bookings (minimum 5 departures):\n", top_3_advance_departure_cities)

# 7. Plot a heatmap displaying correlations of the numerical column and report which pair of numerical columns in the bookings dataset, have the maximum correlation?
numerical_cols = bookings_df.select_dtypes(include=['number'])
correlation_matrix = numerical_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Columns in Bookings Data')
plt.show()

# Find the pair with the maximum correlation (excluding self-correlation)
upper_triangle = correlation_matrix.where(pd.np.triu(pd.np.ones(correlation_matrix.shape), k=1).astype(bool))
max_correlation = upper_triangle.unstack().sort_values(ascending=False).drop_duplicates().head(1)

print("\nPair of numerical columns with the maximum correlation:\n", max_correlation)

# 8. For each service, which is the most used device type for making bookings on the platform?
most_used_device = bookings_df.groupby('service_name')['device_type_used'].apply(lambda x: x.mode()[0])
print("\nMost used device type for each service:\n", most_used_device)

# 9. Plot the trends at a quarterly frequency for the number of bookings by each of the device types, that is, plot a time series for each year and quarter showing the number of bookings performed by each device type
bookings_df['booking_quarter'] = bookings_df['booking_time'].dt.to_period('Q')
quarterly_bookings = bookings_df.groupby(['booking_quarter', 'device_type_used']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
for device in quarterly_bookings.columns:
    plt.plot(quarterly_bookings.index.to_timestamp(), quarterly_bookings[device], label=device)

plt.xlabel('Time (Quarterly)')
plt.ylabel('Number of Bookings')
plt.title('Quarterly Trends of Bookings by Device Type')
plt.legend(title='Device Type')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. Consider the following example and answer the questions related to oBSR:
# We don't have search data, so we cannot directly calculate oBSR.
# We can, however, look at booking frequency per session as a related metric.

# Assuming you have a way to link searches to bookings (e.g., through session_id or customer_id)
# For this example, we'll calculate overall ratios based on monthly and daily aggregations.

# Merge searches and bookings data (you might need to adjust the join key)
merged_df = pd.merge(sessions_df, bookings_df, on='booking_id', how='left')

# Calculate monthly oBSR
merged_df['booking_month'] = pd.to_datetime(merged_df['booking_time']).dt.to_period('M')
monthly_searches = merged_df.groupby('booking_month')['search_id'].nunique()
monthly_bookings = merged_df.groupby('booking_month')['booking_id'].nunique().fillna(0) # Fill NaN bookings with 0
monthly_obsr = monthly_bookings / monthly_searches
print("\nAverage oBSR for each month:\n", monthly_obsr)

# Calculate daily oBSR
merged_df['booking_date'] = pd.to_datetime(merged_df['booking_time']).dt.date
daily_searches = merged_df.groupby('booking_date')['search_id'].nunique()
daily_bookings = merged_df.groupby('booking_date')['booking_id'].nunique().fillna(0) # Fill NaN bookings with 0
daily_obsr = daily_bookings / daily_searches
print("\nAverage oBSR for each day of the week:\n", daily_obsr.groupby(daily_obsr.index.day_name()).mean())

# Plot time series of daily oBSR
plt.figure(figsize=(12, 6))
plt.plot(daily_obsr.index, daily_obsr.values)
plt.xlabel('Date')
plt.ylabel('oBSR (Bookings / Searches)')
plt.title('Time Series of Daily Overall Booking to Search Ratio (oBSR)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
