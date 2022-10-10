# pip install pandas
import pandas as pd
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install scikit-learn
from sklearn.linear_model import LogisticRegression

# Load the CSV
dataset = pd.read_csv('flu_covid_colds_activity.csv')

# Graph
plt.ylabel('Prognosis')
plt.xlabel('Temperature')
plt.scatter(dataset.temperature, dataset.prognosis)
#plt.show()
plt.ylabel('Prognosis')
plt.xlabel('Num People at Home')
plt.scatter(dataset.num_people_at_home, dataset.prognosis)
#plt.show()

# Convert strings to numeric
dataset.body_aches = dataset.body_aches.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.runny_nose = dataset.runny_nose.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.working_at_home = dataset.working_at_home.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.covid_vaccinated = dataset.covid_vaccinated.replace(to_replace=['no', 'yes'], value=[0, 1])
dataset.prognosis = dataset.prognosis.replace(to_replace=['flu', 'covid', 'colds'], value=[0, 1, 2])

# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[[
	'temperature',
	'body_aches',
	'runny_nose',
	'num_people_at_home',
	'working_at_home',
	'covid_vaccinated'
]], dataset.prognosis)

# Test the model
test_temperature = 39
test_body_aches = 'yes'
test_runny_nose = 'yes'
test_num_people_at_home = 3
test_working_at_home = 'no'
test_covid_vaccinated = 'no'

# Convert the test values to numeric
test_body_aches = 1 if test_body_aches == 'yes' else 0
test_runny_nose = 1 if test_runny_nose == 'yes' else 0
test_working_at_home = 1 if test_working_at_home == 'yes' else 0
test_covid_vaccinated = 1 if test_covid_vaccinated == 'yes' else 0

output = model.predict_proba([[
	test_temperature,
	test_body_aches,
	test_runny_nose,
	test_num_people_at_home,
	test_working_at_home,
	test_covid_vaccinated
]])
print("FLU\t", "{:.4f}".format(output[0][0]))
print("COVID\t", "{:.4f}".format(output[0][1]))
print("COLDS\t", "{:.4f}".format(output[0][2]))