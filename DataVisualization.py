import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="Data Visualization", page_icon = ":bar_chart:", layout='wide')
st.markdown('<style>div.block-container{padding-top:10px;}<style>', unsafe_allow_html=True)
st.title("AI Global Index")

df = pd.read_csv("AI_Global_Index.csv")

col1, col2, col3 = st.columns(3)

with col1:
    st.title("Overview")
    st.markdown("""
    <div style="margin-top: 80px;">
    
    </div>
""", unsafe_allow_html=True)
    top_10_countries = df.sort_values('Total score', ascending=False)[['Country', 'Total score']][:10]
    st.dataframe(top_10_countries.style.background_gradient(cmap='seismic'), height=388, width=500)

with col2:
    st.markdown("""
    <div style="display: flex;">
        
    </div>
""", unsafe_allow_html=True)
    title = 'Number of countries by region, cluster, and income group'
    st.markdown(f"<h2 style='font-size: 24px;'>{title}</h2>", unsafe_allow_html=True)
    agg_data = df[['Region', 'Cluster', 'Income group']].groupby(['Region', 'Cluster', 'Income group']).size().reset_index(name='Count')
    path = ['Region', 'Cluster', 'Income group']
    fig = px.sunburst(agg_data, values='Count', path=path, color="Region", height=600, 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )   
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.title("Top 10 Countries")
    selected_score = st.selectbox('Select Score Category:',
                                ['Talent', 'Infrastructure', 'Operating Environment',
                                'Research', 'Development', 'Government Strategy',
                                'Commercial', 'Total score'])
    # Function to create and display top 10 bar chart based on selected score category
    def plot_top_10_countries(df, score_column, score_title):
        top_10_countries = df.nlargest(10, score_column)
        fig = px.bar(
            top_10_countries,
            x=score_column,
            y='Country',
            orientation='h',
            text=score_column,
            title=f'Top 10 Countries Based on {score_title}',
            color='Country'
        )
        fig.update_layout(
            xaxis_title=f'{score_title} Score',
            yaxis_title='Country',
            yaxis_categoryorder='total ascending'
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Country" in df.columns:
        plot_top_10_countries(df, selected_score, selected_score)

col4, col5 = st.columns(2)

with col4: 
    sns.set_theme(style="darkgrid")
    plt.style.use('dark_background')
    # Your code for generating the heatmap
    d = df[['Talent', 'Infrastructure', 'Operating Environment', 'Research', 'Development', 'Government Strategy', 'Commercial', 'Total score']]
    cor = d.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(cor, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cor, mask=mask, cmap=cmap, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Display the plot
    st.pyplot(fig)

with col5:
    # Function to create scatter_geo plot based on selected score category
    def plot_geo_map(df, selected_score):
        fig = px.scatter_geo(
            df,
            locations="Country",
            locationmode='country names',
            color=selected_score,
            size=selected_score,
            hover_name="Country",
            range_color=[df[selected_score].min(), df[selected_score].max()],
            projection="natural earth",
            color_continuous_scale="portland_r",
            #title_world=f'AI {selected_score} Across the Globe'
        )
        title_world=f'AI {selected_score} Across the Globe'
        st.markdown(f"<h2 style='font-size: 30px;'>{title_world}</h2>", unsafe_allow_html=True)
        fig.update_layout(
            showlegend=True,
            width=900,
            height=500,
            autosize=False,
            margin=dict(t=40, b=0, l=5, r=5),
            template="plotly_dark",
        )
        st.plotly_chart(fig)
    if "Country" in df.columns:
        plot_geo_map(df, selected_score)
