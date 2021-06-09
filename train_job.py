from uv_model import read_data as rd
data = rd.LoadDataTrain(rd.LoadDataTrain.getFilePath())
data = data.getData()

from uv_model import preprocess as pp
preprocess = pp.Preprocess(pp.Preprocess.getCountry(),pp.Preprocess.getFeature(),pp.Preprocess.getTarget())
data = preprocess.filter_country(data)
data = data.fillna(0)
X, y = preprocess.splitXy_bydaydiff(data, 5, 45)
X = preprocess.groupbyid(X).drop(columns = 'resettable_device_id_or_app_instance_id')
y = preprocess.groupbyid(y).drop(columns = 'resettable_device_id_or_app_instance_id')
X_train, X_valid, y_train, y_valid = preprocess.split(X, y)

from uv_model import model as md
train = md.Train(X_train, X_valid, y_train, y_valid)
params = train.best()
params['max_depth'] = int(params['max_depth'])
train.performance(params)
my_model = train.fit(params)
# train.save_model(my_model, train.getfilename())