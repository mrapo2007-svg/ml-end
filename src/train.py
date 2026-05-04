from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

models = {
    'k-NN': (
        Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
        {'clf__n_neighbors': [3, 5, 7, 11], 'clf__metric': ['euclidean', 'manhattan']}
    ),
    'Decision Tree': (
        Pipeline([('clf', DecisionTreeClassifier(random_state=42))]),
        {'clf__max_depth': [3, 5, 10, None], 'clf__min_samples_split': [2, 5, 10]}
    ),
    'Random Forest': (
        Pipeline([('clf', RandomForestClassifier(random_state=42))]),
        {'clf__n_estimators': [50, 100], 'clf__max_depth': [5, 10, None]}
    ),
    'SVM': (
        Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))]),
        {'clf__C': [0.1, 1, 10], 'clf__kernel': ['rbf', 'linear']}
    ),
}

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

trained = {}
for name, (pipeline, param_grid) in models.items():
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    gs.fit(X_tv, y_tv)
    trained[name] = gs.best_estimator_
    print(f'{name}: best params = {gs.best_params_}, CV F1 = {gs.best_score_:.4f}')