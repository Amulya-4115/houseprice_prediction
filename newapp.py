import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("🏠 House Price Prediction")

st.header("Step 1: Load and Display Data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

        if "Area" not in data.columns or "Price" not in data.columns:
            st.error("CSV must contain 'Area' and 'Price' columns.")
        else:
            st.header("Step 2: Visualize Area vs Price")
            fig, ax = plt.subplots()
            ax.scatter(data["Area"], data["Price"], color='green')
            ax.set_xlabel("Area (sq ft)")
            ax.set_ylabel("Price (in lakhs)")
            ax.set_title("Scatter Plot: Area vs Price")
            st.pyplot(fig)

            st.header("Step 3: Train Linear Regression Model")
            X = data[["Area"]]
            y = data["Price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully!")

            st.header("Step 4: Visualize Regression Line")
            fig2, ax2 = plt.subplots()
            ax2.scatter(X, y, color='green', label="Actual Data")
            ax2.plot(X, model.predict(X), color='red', label="Regression Line")
            ax2.set_xlabel("Area (sq ft)")
            ax2.set_ylabel("Price (in lakhs)")
            ax2.legend()
            st.pyplot(fig2)

            st.header("Step 5: Predict House Price")
            area_input = st.number_input("Enter Area (in sq ft):", min_value=100.0, step=10.0)
            if st.button("Predict Price"):
                prediction = model.predict([[area_input]])
                st.success(f"💰 Predicted Price: ₹{float(prediction[0]):,.2f} lakhs")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file with Area and Price columns to get started.")
