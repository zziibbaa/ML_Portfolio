import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV


from preprocessing import preprocess_data

train_data , train_label , val_data , val_label ,  test_data , test_label , preprocesser=preprocess_data()

def training_logestic(train_data=train_data, train_label=train_label, preprocesser=preprocesser):
    log_model=LogisticRegressionCV( cv=10  , max_iter=1000 , scoring='f1', random_state=101)
    log_model.fit(train_data , train_label)

    log_model.C_
    log_model.get_params()

    feature_names = preprocesser.get_feature_names_out()
    coefs=(pd.Series(index=feature_names , data=log_model.coef_[0])).sort_values()
    plt.figure(figsize=(10,6))
    sns.barplot(x=coefs.index , y=coefs.values , palette='viridis' , hue=coefs.values)
    plt.tick_params(rotation=90)
    
    return log_model ,coefs


   