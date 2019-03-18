from  fastai.vision import *
import pandas as pd


PATH = 'd:/data_science'
DATAPATH = f'{PATH}/cell_images/'
files = get_files(DATAPATH, extensions='.png', recurse=True)

def get_label(file_path):
    return 'infected' if 'Parasitized' in str(file_path) else 'clean'

def data_slice():
    bs = 64 #Batch size
    data = ImageDataBunch.from_name_func(f'{DATAPATH}', fnames=files,
                                         label_func=get_label,
                                         bs=bs,
                                         ds_tfms=get_transforms(),
                                         size=170).normalize(imagenet_stats)
    df = pd.DataFrame(data.y.items)
    df['category'] = df[0].replace({0: data.classes[0], 1: data.classes[1]})
    print(data.y.items.sum())
    data.show_batch(rows=3, figsize=(7, 6))
    plt.show()
    return data

#ResNet-34模型
def ResNet_34(data):
    #stage1
    learn = create_cnn(data, models.resnet34, pretrained=True, metrics=accuracy)
    print(learn.fit_one_cycle(8))
    #stage2
    # learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot()
    # learn.fit_one_cycle(4, max_lr=slice(1e-6, 1e-5))

if __name__ == '__main__':
    data = data_slice()
    ResNet_34(data)