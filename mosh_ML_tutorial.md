import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# import joblib 
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X,y)

tree.export_graphviz(model, out_file='music-recommender.dot', 
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)

# joblib.dump(model, 'music-recommender.joblib')
# jb.load('music-recommender.joblib')
            
# predictions = model.predict([[21,1]])
