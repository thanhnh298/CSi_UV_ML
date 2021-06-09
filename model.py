from uv_model.preprocess import PreprocessInput as ppi
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from hyperopt.pyll.base import scope
import pickle

class Train:
    space = {}
    model_file_name = 'uv_model/model_file/finalized_model.sav'

    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.X_train  = X_train
        self.X_valid  = X_valid
        self.y_train  = y_train
        self.y_valid  = y_valid
    @classmethod
    def getfilename(cls):
        return cls.model_file_name
    @classmethod
    def getSpace(cls):
        cls.space = {'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.choice('n_estimators', range(1000,10000)),
        'seed': 0}
        return cls.space
    def get_status(self, model):
        space = self.getSpace()

        evaluation = [(self.X_train, self.y_train), ( self.X_valid, self.y_valid)]
        model.fit(self.X_train, self.y_train,
             eval_metric="mae",
             eval_set = evaluation,
             early_stopping_rounds=10,
             verbose = False
             )
        pred = model.predict(self.X_valid)
        mae = mean_absolute_error(self.y_valid, pred)
        return {'loss':mae, 'status': STATUS_OK, 'model': model}
    def obj_fn(self, params):
        model = xgb.XGBRegressor(**params)
        return (self.get_status(model))

    def best(self):
        trials = Trials()
        best = fmin(fn = self.obj_fn,
            space = self.getSpace(),
            algo = tpe.suggest,
            max_evals = 100,
            trials = trials,
            rstate = np.random.RandomState(0)
            )
        
        return best

    def performance(self, params):
        my_model = xgb.XGBRegressor(**params)
        evaluation = [(self.X_train, self.y_train), ( self.X_valid, self.y_valid)]
        my_model.fit(self.X_train, self.y_train,eval_metric="mae",
            eval_set = evaluation,
            early_stopping_rounds=10,
            verbose = False)
        preds = my_model.predict(self.X_valid)
        score = mean_absolute_error(preds, self.y_valid)
        return print('MAE: ', score)
    
    def fit(self, params):
        my_model = xgb.XGBRegressor(**params)
        evaluation = [(self.X_train, self.y_train), ( self.X_valid, self.y_valid)]
        my_model.fit(self.X_train, self.y_train,eval_metric="mae",
            eval_set = evaluation,
            early_stopping_rounds=10,
            verbose = False)
        return my_model

    def save_model(self, my_model, filename):
        pickle.dump(my_model, open(filename, 'wb'))

    

class Predict:
    file_name = Train.getfilename()
    def __init__(self, X):
        self.X = X
    
    def open_model(self):
        self.my_model = pickle.load(open(self.file_name, 'rb'))
        return self.my_model

    def pred(self, id):
        predicts = self.my_model.predict(self.X)
        output = pd.DataFrame({'resettable_device_id_or_app_instance_id':id, 'user_value':predicts})
        return output.to_csv('uv_model/output/output.csv')

