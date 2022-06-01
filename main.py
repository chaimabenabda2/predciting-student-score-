#importing libreries
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as stt
#making the interface
stt.write('''# Application pour la prévision  du score d'etudiant
cette application predit le score d'un éleve a l'aide de nombre des heurs non etudieé
''')
stt.sidebar.header("les parametres d'entrée")
#take the user's variables
def user_input():
    hours=stt.sidebar.slider("heurs d'etude",1.1,9.2,5.0)
    data={"heurs d'etude":hours}
    hour=pd.DataFrame(data,index=[0])
    return hour
df=user_input()
#showing the user's variabels on the other side
stt.subheader("on veut trouver le score pour ce nombres d'heurs")
stt.write(df)

# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
#visiulaztion of the relation between  score and hours of study
fig, ax = plt.subplots()
ax.scatter(data.Hours,data.Scores)
plt.title("score vs hours")
plt.xlabel("hours of study")
plt.ylabel("score of student")
stt.pyplot(fig)
#data.describe()
#s_data.head(10)
# Plotting the distribution of scores
#s_data.plot(x='Hours', y='Scores', style='o')
#plt.title('Hours vs Percentage')
#plt.xlabel('Hours Studied')
#plt.ylabel('Percentage Score')
#plt.show()
#make inputs and output
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
#split this data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
#Training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Training complete.")
#prediction
prediction=regressor.predict(df)
stt.subheader("le score pour cette eléve est:")
stt.write(prediction)
#Mean Squared Error
y_pred=regressor.predict(X_test)
from sklearn import metrics
mse=metrics.mean_absolute_error(y_test,y_pred)
stt.subheader("ce modele predit le score avec une erreur moyenne quadratique:")
stt.write(mse)
#accuracy
aqu=regressor.score(X_test,y_test)
stt.subheader("ce modele predit le score avec une précision:")
stt.write(aqu)
#corolation between dataset variabels
print(data.corr())




