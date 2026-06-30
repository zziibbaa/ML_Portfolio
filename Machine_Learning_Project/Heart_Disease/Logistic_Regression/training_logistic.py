import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline



def training_logistic(train_data , train_label,preprocessor):

    pipeline=Pipeline(steps=[('preprocessor',preprocessor) ,
                             ('classifier' ,LogisticRegressionCV( cv=10  , max_iter=1000 , scoring='f1', random_state=101))])
    pipeline.fit(train_data , train_label)
    
    feature_names = pipeline['preprocessor'].get_feature_names_out()
    coefs=pd.Series(index=feature_names , data=pipeline['classifier'].coef_[0]).sort_values()
    
    fig,ax=plt.subplots(figsize=(10,6))
    sns.barplot(x=coefs.index , y=coefs.values , palette='viridis' , hue=coefs.values , ax=ax)
    plt.tick_params(rotation=90)
    ax.set_title("Logistic Regression Coefficients")
    ax.set_xlabel("Features")
    ax.set_ylabel("Coefficient")
    plt.tight_layout()
    
    return pipeline ,coefs , fig
