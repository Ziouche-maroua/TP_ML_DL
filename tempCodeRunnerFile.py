import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error

data = {
    'packet_size' : [200, 450, 300, 700, 120, 1000, 150, 400, 800, 130,],
    'connection_time' : [30, 50, 25, 80, 10, 100, 15, 45, 90, 12],
    'malicious' : [0, 0, 0, 1, 0, 1, 0, 0, 1, 0] # 0 = Normal , 1= attack
}
df = pd.DataFrame(data)
print(df)

x = df[['packet_size', 'connection_time']]
y = df[['malicious']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42) #30% for testing    random state to divise into same splits
scaler = StandardScaler()  # to normlize differnet scale
x_train_scaled = scaler.fit_transform(x_train)   #scaler.fit alculates mean+standar deviation    Learn scaling parameters (mean/std) from training data and apply them
x_test_scaled = scaler.transform(x_test)    #Use the same scaling parameters on test data

#train KNN and compute MEA
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k , metric = 'manhattan')
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"k={k}, MAE={mae:.2f}")