#Importing the Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import warnings
import sys
import streamlit as st
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)
import streamlit as st
import plotly.express as px

st.set_page_config(layout="centered", page_title="Customer Segmentation")

def load_data():
    data = pd.read_csv('marketing_campaign.csv', sep='\t')
    return data

def data_cleaning(data):
    # Drop NA values
    data = data.dropna()
    
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format='%d-%m-%Y')
    dates = []
    for i in data["Dt_Customer"]:
        i = i.date() # to remove the time stamp, example: 2012-09-04 00:00:00 to 2012-09-04
        dates.append(i)
        
    #Created a feature "Customer_For"
    days = []
    d1 = max(dates) #taking it to be the newest customer
    for i in dates:
        delta = d1 - i
        days.append(delta)
    data["Customer_For"] = days
    data["Customer_For"] = data["Customer_For"].dt.days

    #Feature Engineering
    #Age of customer today 
    data["Age"] = 2021-data["Year_Birth"]

    #Total spendings on various items
    data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

    #Deriving living situation by marital status"Alone"
    data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

    #Feature indicating total children living in the household
    data["Children"]=data["Kidhome"]+data["Teenhome"]

    #Feature for total members in the householde
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

    #Feature pertaining parenthood
    data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

    #Segmenting education levels in three groups
    data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

    #For clarity
    data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

    #Dropping some of the redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)

    return data
    
def data_cleaning2(data):
    # Dropping the outliers by setting a cap on Age and income. 
    data = data[(data["Age"]<90)]
    data = data[(data["Income"]<600000)]
    return data
    
def data_preprocessing(data):    
    #Get list of categorical variables
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    
    #Label Encoding the object dtypes.
    LE=LabelEncoder()
    for i in object_cols:
        data[i]=data[[i]].apply(LE.fit_transform)
        
    #Creating a copy of data
    ds = data.copy()
    # creating a subset of dataframe by dropping the features on deals accepted and promotions
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    #Scaling
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns)
    
    return data, scaled_ds

def dimensionality_reduction(scaled_ds):
    pca = PCA(n_components=3)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
    
    return PCA_ds
    
st.header("Customer Clustering Analysis")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Introduction", "Data", "Cleaning & Preprocessing", "PCA & Clustering", "Clusters Analysis", "Profiling & Conclusion", "About Me"])

with tab1: ## Introduction
    st.header("Grocery firm")
    st.image('img/customer.png')
    import streamlit as st

    col1, col2 = st.columns(2)

    with col1:
        st.write("<p style='text-align: justify;'>This application goes through a clustering analysis using customer data extracted from a grocery company's database. Customer segmentation entails grouping customers based on common characteristics within each cluster. The objective is to pinpoint distinct customer segments, facilitating the implementation of targeted and optimized marketing strategies. Concentrating on specific clusters allows the business to better address the unique needs and behaviors of customers within those groups. This segmentation approach not only increases the significance of each customer to the business but also enhances the effectiveness of advertising campaigns by addressing the specific concerns of different customer types</p>", unsafe_allow_html=True)

    with col2:
        st.image('img/clusters.png')

with tab2: ## Data                
    st.image('img/data.png')
    st.subheader("Original Dataset")  
    st.dataframe(load_data())

with tab3: ## Cleaning & Preprocessing
    data = load_data()
    data = data_cleaning(data)
    
    st.subheader("First cleaning steps:", divider=True)
    st.markdown("""
                - Dropped few NA values
                - Converted the date column to datetime format
                - Created the following features:
                    - "Customer_For" to get the number of days since the customer joined the company
                    - "Age" to get the age of the customer today
                    - "Spent" to get the total spendings on various items
                    - "Living_With" to get the living situation by marital status
                    - "Children" to get the total number of children living in the household
                    - "Family_Size" to get the total number of members in the household
                    - "Is_Parent" to get a binary indicator of parenthood
                    - "Education" to group the education levels in three groups
                - Dropped some of the redundant features
                """)
    
    st.subheader("Outliers Analysis", divider=True)
    # Boxplot in plotly
    st.write("By selecting a feature, you can see the outliers in the boxplot")
    option = st.selectbox('Select a feature:', ('Income', 'Age', 'Spent', 'Customer_For', 'Family_Size'))
    fig = px.box(data, y=option, notched=True, width=500)
    # Plot!
    st.plotly_chart(fig)
    
    data = data_cleaning2(data)
    st.markdown("""
                - The outliers based on Income and Age were dropped
                """)
    
    st.subheader("Clean Dataset", divider=True)  
    st.dataframe(data)
    
    st.subheader("Data Preprocessing steps for PCA & Clustering:", divider=True)    
    st.markdown("""
                - Label encoding the categorical features
                - Scaling the features using the standard scaler
                - Creating a subset dataframe for dimensionality reduction
                """)
    data = data_preprocessing(data)
   
with tab4: ## Dimensionality Reduction & Clustering
    data = load_data()
    data = data_cleaning(data)
    data = data_cleaning2(data)
    data, scaled_ds = data_preprocessing(data)

    st.subheader("Explained variance", divider=True)
    # Plotting with plotly
    pca = PCA()
    pca.fit(scaled_ds)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    fig = px.line(y=exp_var_cumul*100)
    fig.add_trace(px.scatter(y=exp_var_cumul*100).data[0])
    # xlabel and ylabel
    fig.update_xaxes(title_text='Number of Principal Components')
    fig.update_yaxes(title_text='Cumulative Explained Variance')
    # Size and title
    fig.update_layout(
        yaxis=dict(tickmode='array',
            tickvals=list(range(0, 100, 10))),
        width=800,
        height=500,
        title_text='Explained Variance by Number of Principal Components'
    )
    # Set the y-axis limits
    fig.update_yaxes(range=[0, 105])  # Adjust the range as needed
    st.plotly_chart(fig)
    st.markdown("""
                - To be able to able to plot in 3D, three components were selected
                - For the final clustering, six components were selected, those explain more than 70% of the variance.
                """)
    
    # Taking 3 components to be able to plot in 3D
    pca = PCA(n_components=3, random_state=42)
    pca.fit(scaled_ds)
    PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["PC1","PC2", "PC3"]))

    st.subheader("Optimal number of clusters", divider=True)
    agree = st.checkbox('Include trainging timings')
    # Elbow method for Agglomerative Clustering
    model = AgglomerativeClustering()
    visualizer = KElbowVisualizer(model, k=(2,10), timings=agree)
    visualizer.fit(PCA_ds)
    visualizer.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.markdown("""
                - Four clusters were selected based on the elbow method
                """)
    
    #Initiating the Agglomerative Clustering model 
    AC = AgglomerativeClustering(n_clusters=4)
    # fit model and predict clusters
    yhat_AC = AC.fit_predict(PCA_ds)
    PCA_ds["Clusters"] = yhat_AC
    #Adding the Clusters feature to the orignal dataframe.
    data["Clusters"]= yhat_AC
    
    st.subheader("3D Clusters plot", divider=True)
    # Plot with plotly
    fig = px.scatter_3d(PCA_ds, x='PC1', y='PC2', z='PC3', color=data["Clusters"])
    # Reducing opacity and point size
    fig.update_traces(marker=dict(size=2,opacity=0.8))
    # Adding title
    fig.update_layout(title_text='Data points by cluster in 3D (PCs basis)')
    st.plotly_chart(fig)
      
with tab5: ## Clusters Analysis
    # Plot with plotly distribution of clusters
    data = data.sort_values(by=['Clusters']) # To have the labels in the plot 0 to 3
    fig = px.histogram(data, x="Clusters", color="Clusters", width=800, height=500)
    # Adding title
    st.subheader("Clusters distribution", divider=True)
    fig.update_layout(xaxis=dict(tickmode='array',
                                 tickvals=list(range(int(min(data['Clusters'])), int(max(data['Clusters'])) + 1))
                                 )
    )
    st.plotly_chart(fig)
    st.markdown("""
                - The clusters are well balanced
                """)
    
    st.subheader("Promotions accepted", divider=True)
    ##Creating a feature to get a sum of accepted promotions 
    data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]    
    fig = px.histogram(data, x="Total_Promos", color="Clusters", barmode='group', width=800, height=500)
    # Adding title 
    fig.update_layout(title_text='Distribution of the total promotions accepted by cluster')
    st.plotly_chart(fig)
    st.markdown("""
                - Cluster 0 and 2 are the ones that accepted the most promotions accepted
                """)
    
    st.subheader("Deals purchased", divider=True)
    # Distribution NumDealsPurchases by cluster KDE
    fig = px.violin(data, y="NumDealsPurchases", x="Clusters", color="Clusters", box=True, points="all", width=800, height=500)
    # Adding title
    fig.update_layout(title_text=f'Distribution of the NumDealsPurchases by cluster')
    st.plotly_chart(fig)
    st.markdown("""
                - Cluster 0 is the one that purchased the most deals
                """)
    st.write("<h4>Takeaway! The firm should focus on the customers in cluster 0 since they are the ones that accept and purchase the most deals</h4>", unsafe_allow_html=True)
        
with tab6: ## Profiling & Conclusion
    st.subheader("Analysis of clusters characteristics", divider=True)
    st.write("By selecting a feature, you can see the distribution of the feature by cluster, and that wuill help us understand the characteristics of each cluster")
    option = st.selectbox(
    'Select a feature',
    ('Income', 'Age', 'Family_Size', 'Children', 'Is_Parent'))
    # Distribution income by cluster KDE
    fig = px.violin(data, y=option, x="Clusters", color="Clusters", box=True, points="all", width=800, height=500)
    # Adding title
    fig.update_layout(title_text=f'Distribution of the {option} by cluster')
    st.plotly_chart(fig)

    st.subheader("Conclusion", divider=True)                    
    st.image('img/profiles.png', caption='Made with Canva') 
    st.write("<h5>Takeaway reminder! The firm should focus on the customers in cluster 0 since they are the ones that accept and purchase the most deals</h5>", unsafe_allow_html=True)

with tab7: ## About Me
    st.image('img/enthusiast.png')
    st.write("Passionate about strategic decision-making through analytics! In my last job experience, I led key projects, unraveling operational and financial success with the Business Driver Model and optimizing marketing ROI using the Marketing Mix Model. Created a competitive analysis app and a Twitter network observation platform. Excited to collaborate on transformative solutions. Outside work, I'm into spending time on sports! 🚀")

