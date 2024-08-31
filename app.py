import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title('Model Deployment: Logistic Regression - Titanic Dataset')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pclass = st.sidebar.selectbox('Passenger Class (Pclass)',(1, 2, 3))
    Sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    Age = st.sidebar.slider('Age', 0, 100, 29)
    SibSp = st.sidebar.slider('Number of Siblings/Spouses Aboard (SibSp)', 0, 8, 0)
    Parch = st.sidebar.slider('Number of Parents/Children Aboard(Parch)', 0, 6, 0)
    Fare = st.sidebar.slider('Fare', 0, 513, 32)
    Embarked = st.sidebar.selectbox('Part of Embarkation(Embarked)', ('C', 'Q', 'S'))

    data = {'Pclass': Pclass,
            'Sex': 1 if Sex == 'male' else 0,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': 0 if Embarked == 'C' else 1 if Embarked == 'Q' else 2}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

titanic = pd.read_csv("Titanic_train.csv")
titanic['Sex'] = titanic['Sex'].map({'male': 1, 'female': 0})
titanic['Embarked'] = titanic['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Fill Missing Values 
titanic['Age'].fillna(titanic['Age'].median(), inplace= True)
titanic['Fare'].fillna(titanic['Fare'].median(), inplace= True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace= True)

X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y = titanic['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

clf = LogisticRegression()
clf.fit(X, Y)

df_scaled = scaler.transform(df)

# Male Prediction
prediction = clf.predict(df_scaled)
prediction_proba = clf.predict_proba(df_scaled)

st.subheader("Predicted Result")
st.write('Survived' if prediction[0] == 1 else 'Did Not Survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)