import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,root_mean_squared_error
from xgboost import XGBRegressor

data=pd.read_csv("water_potability.csv")
print(data.head())
print(data.describe())
'''''
data.hist(bins=10,figsize=(15,10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True),annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correltaion Heatmap")
plt.show()
'''''
features=data.drop(columns=['Potability'])
target=data['Potability']

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=42) 


param_grid = {
    'n_estimators': [98, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.05]
}
model= XGBRegressor()
gridsearch=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',verbose=1)
gridsearch.fit(x_train,y_train)
print("Best Hyperparameters:", gridsearch.best_params_)

y_pred = gridsearch.best_estimator_.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:",np.sqrt(mse))
print("r2_score:",r2_score(y_test,y_pred))
