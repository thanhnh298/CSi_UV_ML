from sklearn.model_selection import train_test_split
import pandas as pd

class Preprocess:
    feature_cols = ['resettable_device_id_or_app_instance_id', 'day_diff',
       'time_in_game_sum', 'battle_end_sum', 'session_sum',
       'app_exception_fatal_sum', 'app_exception_non_fatal_sum',
       'is_imp_ironsrc_sum', 'rv_imp_ironsrc_sum', 'bn_imp_ironsrc_sum',
       'is_ad_value', 'rv_ad_value', 'bn_ad_value', 'ad_value']
    target_cols = ['resettable_device_id_or_app_instance_id', 'day_diff', 'user_value']
    country = 'United States'
    def __init__(self, country, feature_cols, target_cols):
        self.country = country
        self.feature_cols = feature_cols
        self.target_cols = target_cols
    
    @classmethod
    def getCountry(cls):
        return cls.country
    @classmethod
    def getFeature(cls):
        return cls.feature_cols
    @classmethod
    def getTarget(cls):
        return cls.target_cols

    def filter_country(self, data):
        data = data[data.country == self.country]
        return data
    def splitXy_bydaydiff(self, data, fdays, tdays):
        X = data[self.feature_cols]
        y = data[self.target_cols]
        X = X[X.day_diff <= fdays]
        y = y[y.day_diff <= tdays]
        self.X = X.drop(columns = 'day_diff')
        self.y = y.drop(columns = 'day_diff')
        return self.X, self.y
    
    def groupbyid(self, obj):
        obj = obj.groupby('resettable_device_id_or_app_instance_id', as_index = False).agg('sum').reset_index(drop=True)
        return obj

    @staticmethod 
    def filter_daydiff(object, ndays):
        object = object.loc[(object.day_diff <= ndays) & (object.day_diff >= 0)]
        return object
    def split(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, random_state = 0)
        self.X_train, self.X_valid, self.y_train, self.y_valid = X_train, X_valid, y_train, y_valid
        return self.X_train, self.X_valid, self.y_train, self.y_valid

class PreprocessInput(Preprocess):

    def __init__(self, country, feature_cols):
        self.country = country
        self.feature_cols = feature_cols
            
    @classmethod
    def getCountry(cls):
        return cls.country
    @classmethod
    def getFeature(cls):
        return cls.feature_cols
    @classmethod
    def filter_country(cls, obj):
        obj = obj[obj.country == cls.getCountry()]
        return obj
    
    @staticmethod
    def groupbyid(obj):
        obj = obj.groupby('resettable_device_id_or_app_instance_id', as_index = False).agg('sum').reset_index(drop=True)
        return obj
    
    @staticmethod
    def filter_daydiff(obj, ndays):
        obj = obj.loc[(obj.day_diff <= ndays) & (obj.day_diff >= 0)]
        return obj
    
    def preprocessInput(self, data):
        obj = self.filter_country(data)
        obj = obj[self.getFeature()]
        obj = obj.fillna(0)
        obj = self.groupbyid(obj)
        obj = self.filter_daydiff(obj, 5)
        obj = obj.drop(columns = 'day_diff')
        return obj
        
    def getID(self, data):
        X = self.preprocessInput(data)
        id = X.resettable_device_id_or_app_instance_id
        X = X.drop(columns = 'resettable_device_id_or_app_instance_id')
        return id, X