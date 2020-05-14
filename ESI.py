import pandas as pd
import sklearn as sk
import tensorflow as tf
from keras.constraints import maxnorm
from keras.initializers import glorot_uniform
from sklearn import model_selection
from tensorflow import keras
from keras.optimizers import*
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import time
#ładuje dane z pliku cardio_train.csv

pd.set_option('display.max_columns', 500)# to jest tylko do normlnego wyswietlania danych w konsoli
dane=pd.read_csv("cardio_train.csv", header=0, sep=";")
dataframecolumns=dane.columns;
print(dane.head())

NAME ="FIRST_MODEL-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#sprawdzam czy są wartości null

print(dane.isnull().values.any()) # nie wystepuja puste wartosci
print(dane.describe())
#pozbywam się niepotrzebnych danych oraz danych, które w rzeczywistości nie mogą wystąpić
dane.drop(["id"], axis=1)
dane.drop(dane[dane['ap_lo']>dane['ap_hi']].index, inplace=True)
dane.drop(dane[dane['ap_lo']<45].index , inplace=True)
dane.drop(dane[dane['ap_lo']>200].index, inplace=True)
dane.drop(dane[dane['ap_hi']<60].index, inplace=True)
dane.drop(dane[dane['ap_lo']>250].index, inplace=True)
dane.drop(dane[dane['weight']<35].index, inplace=True)
dane.drop(dane[dane['height']>220].index, inplace=True)
dane.drop(dane[dane['height']<100].index, inplace=True)

#przeliczam dane z kolumny age z dni na lata
dane['age'] = dane['age']//365

#lekka normalizacaj danych
skalar= sk.preprocessing.MinMaxScaler()
dane1= skalar.fit_transform(dane)
dane2= pd.DataFrame(dane1, columns=dataframecolumns)
print(dane2.head())

#  rozdzielenie danych
X = dane2.iloc[:,0:11] # zmienne objansiające
Y= dane2.iloc[:,-1] #samo 'cardio' jest '-1' bo cardio na ostatnim miejscu w df

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=1/6, random_state=123)

# tworzę model i kompiluje
def createModel(optimizer='rmsprop'):
    model = Sequential()
    model.add(Dense(128, input_dim=11, activation=tf.nn.relu, kernel_constraint=maxnorm(2)))

    model.add(Dense(32, activation=tf.nn.relu))

    model.add(Dense(8, activation=tf.nn.relu))

    model.add(Dense(1, activation=tf.nn.sigmoid))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model
modelCheck=createModel()
modelCheck.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=50,
    callbacks=[tensorboard])# to ostanti przydaje sie do porownywania val_loss poszczegolnych modeli

wynikTestu= modelCheck.evaluate(x_test,y_test)
print(wynikTestu)

# zapis modelu
modelCheck.save('CardioModel')

#predykcje
new_model =tf.keras.models.load_model('CardioModel')
predictions = new_model.predict(x_test)
predictionsDataFrame = pd.DataFrame(predictions, index=x_test.index)

predictionsDataFrame.columns=["Pr"]# kolumna prawdopodobieństw
#dodaje id 7 tys bo 1/6  * 70tys
predictionsDataFrame["id"]=range(11441)
predictionsDataFrame["Y"]=np.where(predictionsDataFrame["Pr"]>=0.5,"1","0")
prediction=predictionsDataFrame.drop(["Pr","id"], axis=1)
predictionarray=prediction.astype(np.float)

# sprawdzam ile 1,a ile 0 wychwycił model w próbce testowej
ileZer = predictionsDataFrame[['id','Y']].groupby('Y').count()
print(ileZer)
print("xxxxxxxxxxxxxxxxxxxxx")

#sprawdzam ile zer i jedynek jest rzeczywiscie w probce
ileJedynek = pd.DataFrame(y_test, index=y_test.index)
ileJedynek['id']=range(11441)
macierzJedynekZer=ileJedynek[['id','cardio']].groupby('cardio').count()
print(macierzJedynekZer)

# Aby obliczyc dokładność modelu tworzę macierz błędu
matrix1=ileJedynek.drop(['id'], axis=1) # rzeczywiste wartosci
mb=confusion_matrix(matrix1.values,predictionarray)
# dokładnosc modelu

dokladnosc=mb[0,0]/(mb[0,0]+mb[1,0])
print("Dokładność modelu to: "+str(dokladnosc*100)+"%")

# print(np.argmax(predictions[0])) tak można sprawdzic predykcje dla 1 (pacjenta) pozycji w tabeli dane


#ANALIZA MODELU

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

model1=KerasClassifier(build_fn=createModel, verbose=0)

# wprowadzam dane pomocnicze, aby model nie robil sie 10000000000000000000lat
daneTemporalne= dane.sample(n=1000, random_state=1)
x_train1=daneTemporalne.iloc[:,0:11]
y_train1=daneTemporalne.iloc[:,-1]
optimizers = ['adam', 'nadam','relu']
epochs= [10,20,30]
batches= [10,20,30]
param_grid= dict(optimizer=optimizers,epochs=epochs,batch_size=batches)

grid=GridSearchCV(estimator=model1,param_grid=param_grid)

grid_result=grid.fit(x_train1,y_train1)

means= grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))



