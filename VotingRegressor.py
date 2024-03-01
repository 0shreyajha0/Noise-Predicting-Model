import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler

# Load data from Excel file
data = pd.read_excel('your_excel_file.xlsx')

# Features (independent variables)
X = data[['NumberOfStudents', 'NumberOfACBlower', 'NumberOfFans', 'WindSpeedInRoom', 'TemperatureOfRoom', 'HumidityOfRoom', 'ClassroomAreaSize', 'CeilingHeight', 'NumberOfWindows', 'WallThickness', 'NumberOfDoors', 'IndoorPlants', 'DistanceFromRoadsVehicularTraffic', 'DistanceFromExternalNoiseSources', 'Latitude', 'Longitude', 'TeacherTeachingMethod','TeacherMovement', 'StudentsBehavior', 'ClassRoomActivity', 'VentilationSystem', 'DeskAndChairLayout', 'ExternalNoise', 'RoomDividersAndPartitions', 'UseOfSoundMaskingSystems', 'DayTime', 'DoorStatus', 'WindowStatus', 'WindDirection', 'ExternalNoiseSources', 'AcousticTreatment', 'UseOfNoisyEquipment', 'CeilingShape', 'FlooringMaterial', 'WallAndCeilingMaterials', 'ExteriorBuildingMaterials', 'MaterialOfFurniture']]

# Target variable (dependent variable)
y = data['Noise Level']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create base estimators
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
dtr_model = DecisionTreeRegressor(random_state=42)

# Create a VotingRegressor
voting_regressor = VotingRegressor([('linear', linear_model), ('rf', rf_model), ('dtr', dtr_model)])

# Train the VotingRegressor
voting_regressor.fit(X_train, y_train)

# Save the trained models
joblib.dump(linear_model, 'linear_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(dtr_model, 'decision_tree_model.pkl')
joblib.dump(voting_regressor, 'voting_regressor_model.pkl')

# Make predictions on the test set
y_pred_voting = voting_regressor.predict(X_test)

# Evaluate the VotingRegressor
mse = mean_squared_error(y_test, y_pred_voting)
print(f'Mean Squared Error: {mse}')
print('Voting Regressor Mean Absolute Error:', mean_absolute_error(y_test, y_pred_voting))
print('Voting Regressor Mean Squared Error:', mean_squared_error(y_test, y_pred_voting))
print('Voting Regressor Root Mean Squared Error:', mean_squared_error(y_test, y_pred_voting, squared=False))
