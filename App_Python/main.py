import pandas as pd
import streamlit as st
import statsmodels.api as sm

# Design
st.title('Econometrics tool: OLS regression')
st.write("""
Upload your dataset, choose variables, and perform Ordinary Least Squares (OLS) regression.
Analyse results.
""")

# Upload data
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Display data
    st.write("Data Overview")
    st.write(df.head())

    # Variables selection
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # There must be atleast 2 numerical columns to perform regression
    if len(numerical_columns) < 2:
        st.write("Please upload a dataset with atleast 2 numerical columns")
    else:
        dependant_variable = st.selectbox("Select dependant variable", numerical_columns)
        independant_variables = st.multiselect("Select independant variables", numerical_columns)

        if dependant_variable and independant_variables:
            if dependant_variable in independant_variables:
                st.warning("Dependant variable and independant variable must be different")
            else:
                # OLS regression
                X = df[independant_variables]
                Y = df[dependant_variable]

                # We add a constant to the model
                X = sm.add_constant(X)

                # model estimation
                st.write("OLS Regression Results")
                model = sm.OLS(Y, X).fit()
                st.write(model.summary())


st.write("[Github Link](https://github.com/Dam77/EasyOLS)")
            


