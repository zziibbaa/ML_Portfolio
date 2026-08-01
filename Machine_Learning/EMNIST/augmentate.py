import torchvision.transforms as T
def augmentation( *augment_param):
    if augment_param :
        pil_transforms = []
        tensor_transforms = []
        tensor_only = {"RandomErasing"}
        
        for Aug_name , aug_kw in augment_param:
            aug_func=getattr(T , Aug_name)(**aug_kw)
            if Aug_name in tensor_only:
                tensor_transforms.append(aug_func)

            else:
                pil_transforms.append(aug_func)

        transforms = [T.ToPILImage(),
                      *pil_transforms,
                       T.ToTensor(),
                       *tensor_transforms,
                        T.Normalize(mean=(.5,) , std=(.5,))]
        ts=T.Compose(transforms)
        

    else:
        ts=T.Compose([T.ToPILImage(),
                      T.ToTensor(),
                      T.Normalize(mean=(.5,), std=(.5,))])
       

    return ts