import mlflow
from data_loader import load_data

model=mlflow.pyfunc.load_model('models:/HeartDiseasePipeline/latest')
print(model)


data = load_data()
x = data.drop("DISEASE", axis=1)
sample = x.iloc[:5]

pred = model.predict(sample)

print(pred)