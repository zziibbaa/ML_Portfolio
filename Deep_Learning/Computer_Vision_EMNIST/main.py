from data import load_emnist_data, CustomDataClass
from augmentate import augmentation
from Architecture_Model import CNN_model
from model import build_optimizer_scheduler_loss_fun
from train_model import train_model
from evaluate import *
from torch.utils.data import DataLoader


def evaluate_pipeline(model, history,test_loader, device, class_name):
    
    images, y_true, pred = predict_model(test_loader, model, device )

    plot_loss(history)
    plot_accuracy(history)

    classification_report_plot(class_name, y_true, pred)
    plot_confusion_matrix(y_true, pred, class_name)
    show_wrong_predictions(images, y_true, pred, class_name)

    error_df = show_common_errors(class_name, y_true, pred)

    return error_df

def main(epochs , batch_size , optimizer_name ,aug=False, augment_param=None,
         scheduler_name=None ,scheduler_position=None,
         image_chanel=1, n_classes=26,
         chanels=(64,128,256), kernel_size=3, padding=1,
         drop_out_conv=.1 , drop_out_fc=.4 ,
         hidden_unit=(128,64) , use_batchnorm_dropout=True , **kwargs ):

    optimizer_kwargs = kwargs.get("optimizer_kwargs", {})
    scheduler_kwargs = kwargs.get("scheduler_kwargs", {})
    
    #load data
    (train_data , train_label) , (test_data , test_label) , class_name=load_emnist_data()
    
    if aug:
        train_transform=augmentation(*augment_param)
        test_transform=augmentation()

    else:
        train_transform=augmentation()
        test_transform=augmentation()
    # create dataset
    train_dataset=CustomDataClass((train_data,train_label),train_transform)
    test_dataset=CustomDataClass((test_data,test_label),test_transform)

    # create data_loader
    train_loader=DataLoader(train_dataset , batch_size=batch_size , shuffle=True , drop_last=True)
    test_loader=DataLoader(test_dataset , batch_size=batch_size , shuffle=False , drop_last=False)

    if scheduler_name == "OneCycleLR":
        scheduler_kwargs["epochs"] = epochs
        scheduler_kwargs["steps_per_epoch"] = len(train_loader)
    # define model
    model_cnn=CNN_model(image_chanel=image_chanel, n_classes=n_classes,
                        chanels=chanels, kernel_size=kernel_size, padding=padding,
                        drop_out_conv=drop_out_conv , drop_out_fc=drop_out_fc ,
                        hidden_unit=hidden_unit , use_batchnorm_dropout=use_batchnorm_dropout)
    # define optimizer , scheduler , loss_fun
    optimizer , scheduler , loss_fun=build_optimizer_scheduler_loss_fun(model_cnn , optimizer_name ,
                                                                        scheduler_name , optimizer_kwargs ,
                                                                        scheduler_kwargs )
   # train model
    model , history , device=train_model(epochs=epochs ,train_loader=train_loader , test_loader=test_loader ,
                                         model=model_cnn , optimizer=optimizer , loss_fun=loss_fun ,
                                         scheduler=scheduler , scheduler_position=scheduler_position)

    # evaluate model
    error_df = evaluate_pipeline(model, history, test_loader, device, class_name)

    return {"error_df": error_df}
 