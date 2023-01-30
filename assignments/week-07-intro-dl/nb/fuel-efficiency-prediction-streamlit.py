import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from shap_plots import plot_local_explain, plot_global_explain

# Add and resize an image to the top of the app

img_fuel = Image.open("./assignments/week-07-intro-dl/img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

models = ["dnn", "tpot", "linear"]

# Import train dataset to DataFrame
train_df = pd.read_csv("../dat/train_ds.csv", index_col=0)
model_results_df = pd.read_csv("../dat/model_results.csv", index_col=0)
all_shaps = {m: pd.read_csv(f"../dat/shap_{m}.csv", index_col=0) for m in models}

sample_indices = {f"Item {m}":i for i, m in enumerate(train_df.index)}

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )

    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (n for n in models if n != model1_select)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3, tab4 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Local Model Explainability", "👌 Global Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    
    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)
        df1 = model_results_df.loc[model1_select+"_model", :]
        st.dataframe(df1)


    # Build confusion matrix for second model
    with col2:
        st.header(model2_select)
        df2 = model_results_df.loc[model2_select+"_model", :]
        st.dataframe(df2)


with tab3: 
    
    st.selectbox(
        "Sample Number:",
        (sample_indices)
    )
    

    st.header(model1_select)
    st.write(plot_local_explain(all_shaps[model1_select].values.squeeze(), train_df.drop(columns=["MPG"])))
    
  
    st.header(model2_select)
    st.write(plot_local_explain(all_shaps[model2_select].values.squeeze(), train_df.drop(columns=["MPG"])))
    
with tab4:
    st.header(model1_select)
    st.write(plot_global_explain(all_shaps[model1_select].values.squeeze(), train_df.drop(columns=["MPG"])))
    
    st.header(model2_select)
    st.write(plot_global_explain(all_shaps[model2_select].values.squeeze(), train_df.drop(columns=["MPG"])))