import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
### Configure Page
st.set_page_config(page_title="🌐📊The Global Superstore",layout="wide")

###Load data
store = pd.read_csv('cleaned_data.csv',encoding="ISO-8859-1")

### Sidebar
option = st.sidebar.selectbox("Pick a choice:",['Home','EDA','ML'])

###-------------------------Home-------------------------
if option == 'Home':
    st.title("The Global Superstore Dashboard🌐📊")
    st.markdown('✒️Author : Mahmoud Mohamed')
    st.header("👋Welcome to my Global Superstore Dashboard!!")
    st.markdown("""
    This interactive application is designed to explore, analyze, and gain insights from e-commerce data, as well as provide intelligent product recommendations.

    ### 🔍 What you can do:

    - 📊 Analyze customer behavior and sales performance  
    - 🧠 Explore customer segmentation and patterns  
    - 🎯 Predict order priority using machine learning models  
    - 🛒 Get personalized product recommendations  

    Use the sidebar to navigate between different sections of the app and start exploring the data.

    ---

    🚀 *Built to transform data into actionable insights.*
    """)
    st.write("The E-commerce Data:")
    st.dataframe(store.head(5))
###-------------------------EDA--------------------------    
elif option == 'EDA':
    st.title("📈Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section provides an overview of the dataset, helping you understand its structure, distributions, and key patterns.
    """)
    ### ---------------------Layout------------------------------
    col1 , col2 = st.columns(2)
    ###---------------------Line Chart---------------------------
    st.subheader("Sales By Each Quarter")
    quarterly_sales = store.groupby('Quarter')['Sales'].sum().reset_index()
    quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)
    sales_by_quarter = px.line(quarterly_sales, x='Quarter', y='Sales')
    sales_by_quarter.update_yaxes(range=[0,1600000])
    sales_by_quarter.update_xaxes(range=[-0.2,15.5], tickangle=-45)
    sales_by_quarter.update_layout(title='Sales by Each Quarter')
    st.plotly_chart(sales_by_quarter,use_container_width=True,key="sales_quarter")
    st.markdown("#### 🔍 Insight:")
    st.info("This Insight shows that the sales increases as time passes by.But There was always a drop in The First quarter in each year and a rise in The Fourth quarter in each year as well.")
    ###----------------------Bar Chart---------------------------
    st.subheader("Revenue: Discount VS No Discount")
    revenue = store.groupby(store['Discount'] > 0)['Sales'].sum()
    revenue_fig = px.bar(revenue, x=revenue.index, y=revenue.values)
    revenue_fig.update_xaxes(title = 'Discount')
    revenue_fig.update_yaxes(title = 'Revenue')
    revenue_fig.update_layout(title = 'Revenue: Discount VS No Discount')
    st.plotly_chart(revenue_fig,use_container_width=True,key="revenue_discount")
    st.markdown("#### 🔍 Insight:")
    st.info("This Insight shows that the sales increases as time passes by.But There was always a drop in The First quarter in each year and a rise in The Fourth quarter in each year as well.")    
    ###-------------------Pie Chart-------------------------------
    st.subheader("Our Favourite Segment")
    best_selling_segment_pie = px.pie(store.groupby('Segment')['Sales'].sum().sort_values(ascending=False).reset_index(), names='Segment', values='Sales')
    best_selling_segment_pie.update_layout(title='Best Selling Segment')
    st.plotly_chart(best_selling_segment_pie,use_container_width=True,key="best_selling_segment")
    st.markdown("### 🔍 Insight:")
    st.info("This Insight shows The Favourite Customer Segmentation,As shown in The Figure Consumer customers are the most buyers which means our business focuses more on The Individual Buyers")
    ###-------------------Bar Chart--------------------------------
    st.subheader('Top 5 Products')
    best_selling_product = px.bar(store.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).reset_index().head(5), x='Product Name', y='Sales')
    best_selling_product.update_layout(title='Top 5 Products')
    st.plotly_chart(best_selling_product,use_container_width=True,key="best_selling_product")
    st.markdown("### 🔍 Insight:")
    st.info("""
    This Insight shows Top 5 Products in terms of sales.
    As shown in The Figure Smartphones dominates The store.
     Smartphones are not only the most sold products but also contribute significantly to overall revenue.
     """)
###--------------------------ML--------------------------    
elif option == 'ML':
    st.title("🤖 Machine Learning Section")
    st.subheader('Profit prediction for each order')
    st.write('Enter Shipping cost, Quantity , Sales, Discount(if there is)')
    s = st.number_input('Sales')
    sc = st.number_input('Shipping Cost')
    d = st.number_input('Discount( If None enter 0)')
    q = st.number_input('Quantity')
    btn1 = st.button('Predict')
    pp = pickle.load(open('profit_model.pkl','rb'))
    result = pp.predict([[s, sc, d, q]])
    if btn1:
        st.write('The Predicted Profit is :', result[0])
    st.subheader('Product Recommendation')
    st.write('Enter the Product Name')  
    product_name = st.text_input('Product Name')
    btn2 = st.button('Recommend')
    pn = pickle.load(open('product_rec.pkl','rb'))
    if btn2:
        rec_vectorizer = pn['vectorizer']
        rec_model = pn['model']
        rec_data = pn['data']
        product_vec = rec_vectorizer.transform([product_name])
        distances, indices = rec_model.kneighbors(product_vec, n_neighbors=5)
        recommended_products = rec_data.iloc[indices[0]]['text'].values
        st.write("Recommended Products:")
        for i, prod in enumerate(recommended_products):
            st.write(f"{i+1}. {prod}")
    st.subheader('Order Priority Prediction')
    st.write('Enter the Order Details to Predict its Priority')
    ship_mode = st.selectbox('Shipping Mode', ['First Class', 'Same Day', 'Second Class', 'Standard Class'])
    ship_cost = st.number_input('Shipping Cost',key = 'order_priority_shipping_cost')
    segment= st.selectbox('Customer Segment', ['Consumer', 'Corporate', 'Home Office'])    
    order_priority_btn = st.button('Predict Order Priority')
    op_model = pickle.load(open('order_priority_model.pkl','rb'))
    if order_priority_btn:
        ship_mode_mapping = {'First Class': 0, 'Same Day': 1, 'Second Class': 2, 'Standard Class': 3}
        segment_mapping = {'Consumer': 0, 'Corporate': 1, 'Home Office': 2}
        ship_mode_encoded = ship_mode_mapping[ship_mode]
        segment_encoded = segment_mapping[segment]
        order_priority_prediction = op_model.predict([[ship_mode_encoded, ship_cost, segment_encoded]])
        if order_priority_prediction[0] == 0:
            st.write("Predicted Order Priority: Critical")
        elif order_priority_prediction[0] == 1:
            st.write("Predicted Order Priority: High")
        elif order_priority_prediction[0] == 2:
            st.write("Predicted Order Priority: Low")
        elif order_priority_prediction[0] == 3:
            st.write("Predicted Order Priority: Medium")    
        
    