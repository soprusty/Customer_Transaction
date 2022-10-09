import sklearn.neighbors as knn
import streamlit as st
import pandas as pd
import pickle

#Loading up the Regression model we created
model = knn.KNeighborsClassifier()
#model.load_model('finalized_model.sav')

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

#Caching the model for faster loading
@st.cache


# Define the prediction function
def predict(var_0,
var_1	,
var_2	,
var_3	,
var_4	,
var_5	,
var_6	,
var_7	,
var_8	,
var_9	,
var_10	,
var_11	,
var_12	,
var_13	,
var_14	,
var_15	,
var_16	,
var_17	,
var_18	,
var_19	,
var_20	,
var_21	,
var_22	,
var_23	,
var_24):
    
    
    

    prediction = loaded_model.predict(pd.DataFrame([[var_0	,
var_1	,
var_2	,
var_3	,
var_4	,
var_5	,
var_6	,
var_7	,
var_8	,
var_9	,
var_10	,
var_11	,
var_12	,
var_13	,
var_14	,
var_15	,
var_16	,
var_17	,
var_18	,
var_19	,
var_20	,
var_21	,
var_22	,
var_23	,
var_24]], columns=['var_0',
'var_1',
'var_2',
'var_3',
'var_4',
'var_5',
'var_6',
'var_7',
'var_8',
'var_9',
'var_10',
'var_11',
'var_12',
'var_13',
'var_14',
'var_15',
'var_16',
'var_17',
'var_18',
'var_19',
'var_20',
'var_21',
'var_22',
'var_23',
'var_24']))
    return prediction


st.title('Customer Transaction Predictor')
st.image("""https://www.india.com/wp-content/uploads/2014/08/666.jpg""")
st.header('Enter the characteristics of the Customer:')
#carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0)
#cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
#color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
#clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
#depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0)
#table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)
#x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)
#y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)
#z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)

var_0 = st.number_input('var_0:',min_value=0.1, max_value=100.0, value=1.0)
var_1 = st.number_input('var_1:',min_value=0.1, max_value=100.0, value=1.0)
var_2 = st.number_input('var_2:',min_value=0.1, max_value=100.0, value=1.0)
var_3 = st.number_input('var_3:',min_value=0.1, max_value=100.0, value=1.0)
var_4 = st.number_input('var_4:',min_value=0.1, max_value=100.0, value=1.0)
var_5 = st.number_input('var_5:',min_value=0.1, max_value=100.0, value=1.0)
var_6 = st.number_input('var_6:',min_value=0.1, max_value=100.0, value=1.0)
var_7 = st.number_input('var_7:',min_value=0.1, max_value=100.0, value=1.0)
var_8 = st.number_input('var_8:',min_value=0.1, max_value=100.0, value=1.0)
var_9 = st.number_input('var_9:',min_value=0.1, max_value=100.0, value=1.0)
var_10 = st.number_input('var_10:',min_value=0.1, max_value=100.0, value=1.0)
var_11 = st.number_input('var_11:',min_value=0.1, max_value=100.0, value=1.0)
var_12 = st.number_input('var_12:',min_value=0.1, max_value=100.0, value=1.0)
var_13 = st.number_input('var_13:',min_value=0.1, max_value=100.0, value=1.0)
var_14 = st.number_input('var_14:',min_value=0.1, max_value=100.0, value=1.0)
var_15 = st.number_input('var_15:',min_value=0.1, max_value=100.0, value=1.0)
var_16 = st.number_input('var_16:',min_value=0.1, max_value=100.0, value=1.0)
var_17 = st.number_input('var_17:',min_value=0.1, max_value=100.0, value=1.0)
var_18 = st.number_input('var_18:',min_value=0.1, max_value=100.0, value=1.0)
var_19 = st.number_input('var_19:',min_value=0.1, max_value=100.0, value=1.0)
var_20 = st.number_input('var_20:',min_value=0.1, max_value=100.0, value=1.0)
var_21 = st.number_input('var_21:',min_value=0.1, max_value=100.0, value=1.0)
var_22 = st.number_input('var_22:',min_value=0.1, max_value=100.0, value=1.0)
var_23 = st.number_input('var_23:',min_value=0.1, max_value=100.0, value=1.0)
var_24 = st.number_input('var_24:',min_value=0.1, max_value=100.0, value=1.0)





if st.button('Predict Price'):
    target = predict(var_0,
var_1,
var_2,
var_3,
var_4,
var_5,
var_6,
var_7,
var_8,
var_9,
var_10,
var_11,
var_12,
var_13,
var_14,
var_15,
var_16,
var_17,
var_18,
var_19,
var_20,
var_21,
var_22,
var_23,
var_24)
    st.success(f'The predicted price of the diamond is ${target[0]:.2f} USD')
