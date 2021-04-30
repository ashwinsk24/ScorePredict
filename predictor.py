### Custom definitions and classes if any ###
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib



def predictRuns(input_test):

    with open('regression_model.joblib','rb') as f:
        regressor = joblib.load(f)

    with open('venue_encoder.joblib','rb') as f:
        venue_encoder = joblib.load(f) 
    
    with open('team_encoder.joblib','rb') as f: 
        team_encoder = joblib.load(f)

 # read test data
    test_case = pd.read_csv(input_test)

 # encode venue and batting and bowling
    test_case['venue'] = venue_encoder.fit_transform(test_case['venue'])
    test_case['batting_team'] = team_encoder.fit_transform(test_case['batting_team'])
    test_case['bowling_team'] = team_encoder.fit_transform(test_case['bowling_team'])


 # make sure that the order of columns is same as that fed to model 
    test_case = test_case[['venue','innings','batting_team','bowling_team']]


 # convert input test case into numpy array 
    testArray = test_case.to_numpy()


 #Encode venue, batting and bowling teams 
    test_case = np.concatenate((np.eye(42)[testArray[:,0]],
                            np.eye(2)[testArray[:,1] -1 ], 
                            np.eye(15)[testArray[:,2]],
                            np.eye(15)[testArray[:,3]],
                            ),
                            axis = 1) 


    return regressor.predict(test_case)







#Importing the dataset
with open('all_matches.csv') as f:
    ipl_data = pd.read_csv(f)
 
relevantColumns = ['match_id','venue','innings','ball','batting_team','bowling_team','striker','non_striker','bowler','runs_off_bat','extras','wides','noballs','byes','legbyes','penalty']
 
ipl_data = ipl_data[relevantColumns]
 
ipl_data['total_runs']=ipl_data['runs_off_bat']+ipl_data['extras']


ipl_data=ipl_data.drop(columns=['runs_off_bat','extras'])


ipl_data=ipl_data[ipl_data['ball']<=5.6]

ipl_data=ipl_data[ipl_data['innings']<=2]

 
ipl_data = ipl_data.groupby(['match_id',
                             'venue',
                             'innings',
                             'batting_team',
                             'bowling_team']).total_runs.sum()

ipl_data= ipl_data.reset_index()
ipl_data= ipl_data.drop(columns=['match_id'])
ipl_data.to_csv("myPreprocessed.csv", index=False)



#Training the dataset
data = pd.read_csv('myPreprocessed.csv')

venue_encoder = LabelEncoder()
team_encoder = LabelEncoder()

data['venue']=venue_encoder.fit_transform(data['venue'])
data['batting_team']=team_encoder.fit_transform(data['batting_team'])
data['bowling_team']=team_encoder.fit_transform(data['bowling_team'])

anArray = data.to_numpy()

X,y= anArray[:,:3],anArray[:,3]

X = np.concatenate((np.eye(42)[anArray[:,0]],
                                    np.eye(2)[anArray[:,1] -1 ],
                                    np.eye(15)[anArray[:,2]],
                                    np.eye(15)[anArray[:,3]],
                                    ), axis = 1)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the dataset
lin = LinearRegression()
lin.fit(X_train,y_train)

joblib.dump(lin, 'regression_model.joblib')
joblib.dump (venue_encoder, 'venue_encoder.joblib') 
joblib.dump(team_encoder, 'team_encoder.joblib')

print(lin.score(X_test,y_test))





