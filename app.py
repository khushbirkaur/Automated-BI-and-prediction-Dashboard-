import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Automated BI Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated Business Intelligence Dashboard")

# ---------------- AUTO REFRESH ----------------
st_autorefresh(interval=30000, key="datarefresh")

# ---------------- LOAD DATA FROM GOOGLE SHEETS ----------------
@st.cache_data(ttl=300)
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1InuCpmy1sPw9wl-kyD_S4uPsUO1C6UlfXo2BOARq298/export?format=csv"
    df = pd.read_csv(sheet_url)
    return df

df = load_data()

# ---------------- DATA PREPROCESSING ----------------
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month Name"] = df["Order Date"].dt.strftime("%B")
df["Month Year"] = df["Order Date"].dt.strftime("%b %Y")

df["Shipping Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

year = st.sidebar.multiselect(
    "Select Year",
    sorted(df["Year"].unique()),
    default=df["Year"].unique()
)

region = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

state = st.sidebar.multiselect(
    "Select State",
    df["State"].unique(),
    default=df["State"].unique()
)

city = st.sidebar.multiselect(
    "Select City",
    df["City"].unique(),
    default=df["City"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

segment = st.sidebar.multiselect(
    "Select Segment",
    df["Segment"].unique(),
    default=df["Segment"].unique()
)

filtered_df = df[
    (df["Year"].isin(year)) &
    (df["Region"].isin(region)) &
    (df["State"].isin(state)) &
    (df["City"].isin(city)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment))
]

# ---------------- KPI CARDS ----------------
total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["Order ID"].nunique()
avg_sales = filtered_df["Sales"].mean()
avg_shipping = filtered_df["Shipping Time"].mean()

total_customers = filtered_df["Customer ID"].nunique()
total_states = filtered_df["State"].nunique()
total_cities = filtered_df["City"].nunique()
total_products = filtered_df["Product Name"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Orders", total_orders)
col3.metric("Avg Sales", f"${avg_sales:,.2f}")
col4.metric("Avg Shipping Days", f"{avg_shipping:.1f}")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Unique Customers", total_customers)
col6.metric("Total States", total_states)
col7.metric("Total Cities", total_cities)
col8.metric("Total Products", total_products)

st.markdown("---")

# ---------------- SALES TREND ----------------
sales_trend = filtered_df.groupby("Month Year")["Sales"].sum().reset_index()

fig1 = px.line(
    sales_trend,
    x="Month Year",
    y="Sales",
    title="Sales Trend Over Time",
    markers=True
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- REGION ANALYSIS ----------------
region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()

fig2 = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    color="Region",
    title="Sales by Region"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- CATEGORY ANALYSIS ----------------
category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()

fig3 = px.pie(
    category_sales,
    names="Category",
    values="Sales",
    title="Sales by Category"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- TOP CUSTOMERS ----------------
top_customers = (
    filtered_df.groupby("Customer Name")["Sales"]
    .sum()
    .nlargest(10)
    .reset_index()
)

fig4 = px.bar(
    top_customers,
    x="Customer Name",
    y="Sales",
    title="Top 10 Customers"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------- TOP PRODUCTS ----------------
top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .nlargest(10)
    .reset_index()
)

fig5 = px.bar(
    top_products,
    x="Product Name",
    y="Sales",
    title="Top 10 Products"
)

st.plotly_chart(fig5, use_container_width=True)

# ---------------- SHIPPING MODE ----------------
ship_mode = filtered_df.groupby("Ship Mode")["Sales"].sum().reset_index()

fig6 = px.bar(
    ship_mode,
    x="Ship Mode",
    y="Sales",
    color="Ship Mode",
    title="Sales by Shipping Mode"
)

st.plotly_chart(fig6, use_container_width=True)

# ---------------- TOP STATES ----------------
state_sales = (
    filtered_df.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig7 = px.bar(
    state_sales,
    x="State",
    y="Sales",
    title="Top 10 States by Sales"
)

st.plotly_chart(fig7, use_container_width=True)

# ---------------- DATA TABLE ----------------
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

# ---------------- SALES PREDICTION ----------------
st.markdown("---")
st.header("📈 Future Sales Prediction")

yearly_sales = df.groupby("Year")["Sales"].sum().reset_index()

X = yearly_sales["Year"].values.reshape(-1,1)
y = yearly_sales["Sales"].values

model = LinearRegression()
model.fit(X, y)

last_year = yearly_sales["Year"].max()
future_years = np.array([last_year+1, last_year+2, last_year+3]).reshape(-1,1)

predicted_sales = model.predict(future_years)

future_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Sales": predicted_sales
})

st.subheader("Predicted Sales for Upcoming Years")
st.dataframe(future_df)

combined_df = pd.concat([
    yearly_sales.rename(columns={"Sales":"Value"}),
    future_df.rename(columns={"Predicted Sales":"Value"})
])

combined_df["Type"] = ["Actual"]*len(yearly_sales) + ["Predicted"]*len(future_df)

fig_pred = px.line(
    combined_df,
    x="Year",
    y="Value",
    color="Type",
    markers=True,
    title="Actual vs Predicted Sales"
)

st.plotly_chart(fig_pred, use_container_width=True)