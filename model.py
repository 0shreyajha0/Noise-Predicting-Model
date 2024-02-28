import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import DecisionTreeRegressor

# Load data from Excel file
data = pd.read_excel('your_excel_file.xlsx')

# Created data frame
df = pd.DataFrame(data)

# Features (independent variables) #update this
X = df[['Number of Students', 'Teacher Teaching Method', 'Classroom Area', 'Ceiling Height']]
X = df[['NumberOfStudents', 'NumberOfACBlower', 'NumberOfFans', 'WindSpeedInRoom', 'TemperatureOfRoom', 'HumidityOfRoom', 'ClassroomAreaSize', 'CeilingHeight', 'NumberOfWindows', 'WallThickness', 'NumberOfDoors', 'IndoorPlants', 'DistanceFromRoadsVehicularTraffic', 'DistanceFromExternalNoiseSources', 'Latitude', 'Longitude', 'TeacherTeachingMethod','TeacherMovement', 'StudentsBehavior', 'ClassRoomActivity', 'VentilationSystem', 'DeskAndChairLayout', 'ExternalNoise', 'RoomDividersAndPartitions', 'UseOfSoundMaskingSystems', 'DayTime', 'DoorStatus', 'WindowStatus', 'WindDirection', 'ExternalNoiseSources', 'AcousticTreatment', 'UseOfNoisyEquipment', 'CeilingShape', 'FlooringMaterial', 'WallAndCeilingMaterials', 'ExteriorBuildingMaterials', 'MaterialOfFurniture']]


# Target variable (dependent variable)
y = df['Noise Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'sample.py')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))

# Create a Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print('Random Forest Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Random Forest Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print('Random Forest Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf, squared=False))

# Create a Decision Tree Regressor
rf_model = DecisionTreeRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Decision Tree Regressor
print(' Decision Tree Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print(' Decision Tree Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print(' Decision Tree Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf, squared=False))



