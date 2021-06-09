from uv_model import read_data as rd
from uv_model import model as md
from uv_model import preprocess as pp


data = rd.LoadDataInput(rd.LoadDataInput.getFilePath())
data = data.getData()

preprocess = pp.PreprocessInput(pp.PreprocessInput.getCountry(), pp.PreprocessInput.getFeature())
id, X = preprocess.getID(data)

model = md.Predict(X)
my_model = model.open_model()
model.pred(id)
