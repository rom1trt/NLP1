from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def custom_score(gt, y_pred):
    precision = precision_score(gt, y_pred)
    recall = recall_score(gt, y_pred)
    f1 = f1_score(gt, y_pred)
    combined_score = precision + recall + f1
    return combined_score

def grid_search(model, param_grid, scoring, X_train, y_train, cv=5, verbose=1, n_jobs=-1):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    return grid_search

def grid_search_predict(grid_search):
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_params, best_model

def random_search(model, param_distributions, scoring, X_train, y_train, n_iter=10, cv=5, verbose=1, n_jobs=-1):
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs)
    random_search.fit(X_train, y_train)
    return random_search

def random_search_predict(random_search):
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_params, best_model
