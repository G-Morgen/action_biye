from model import tsn, i3d
import models
import dfl

def get_model(name,modality,num_classes,dropout,input_size,input_segments,use_middle_feature=False):

    if 'i3d' not in name and 'dfl' not in name:
        model = models.TSN(base_model=name,modality=modality,num_segments=input_segments,
            num_class=num_classes,dropout=dropout, use_middle_feature=use_middle_feature)
    elif 'dfl' in name:
        model = dfl.DFL(base_model=name.split('_')[1],modality=modality,num_segments=input_segments,
            num_class=num_classes,dropout=dropout)
    elif name == 'i3d_resnet34':
        model = i3d.resnet34(pretrained=True,num_classes=num_classes,modality=modality, input_segments=input_segments, dropout=dropout)
    elif name == 'i3d_resnet50':
        model = i3d.resnet50(pretrained=True,num_classes=num_classes,modality=modality, input_segments=input_segments, dropout=dropout)
    else:
        raise ValueError('no that model architecture')

    return model

if __name__ == "__main__":
    a = get_model('bn_inception','rgb',101,0.5,224,8)
    print(a)