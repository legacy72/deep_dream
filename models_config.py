from models import *


class Model:
    """
    Можно менять различные значения в разных слоях layer_contributions для получения различных результатов
    Первые уровни отвечают за общие признаки. Последние - за абстрактные вещи. Поэтому чтобы добиться более 
    психодолических эффектов, нужно ставить потери на последних уровнях.
    """
    model = None
    model_name = ''
    layer_contributions = {}

    def inception_model(self):
        self.model_name = 'inception'
        self.layer_contributions = {
            'mixed2': 0.2,
            'mixed3': 3.,
            'mixed4': 2.,
            'mixed5': 1.5,
        }
        self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        print(self.model.summary()) # show layer_contribitions

    def vgg16_model(self):
        """
        Моя любимая комбинация
        """
        self.model_name = 'vgg16'
        self.layer_contributions = {
            # 'block1_conv1': 0.2,
            # 'block2_conv1': 0.3,
            # 'block3_conv1': 0.2,
            # 'block4_conv1': 0.3,
            'block5_conv3': 0.2,
        }
        self.model = vgg16.VGG16(weights='imagenet', include_top=False)
        print(self.model.summary()) # show layer_contribitions

    def xception_model(self):
        self.model_name = 'xception'
        self.layer_contributions = {
            # 'add_1': 0.1,
            # 'add_2': 0.2,
            # 'add_3': 0.3,
            # 'add_4': 0.2,
            # 'add_5': 0.5,
            'add_6': 0.2,
            # 'add_7': 0.3,
            # 'add_8': 0.1,
            # 'add_9': 0.3,
        }
        self.model = xception.Xception(weights='imagenet', include_top=False)
        print(self.model.summary()) # show layer_contribitions

    def resnet50_model(self):
        self.model_name = 'resnet50'
        self.layer_contributions = {
            # 'res2a_branch2a': 0.2,
            # 'res2c_branch2c': 0.6,
            # 'res4a_branch2a': 0.7,
            # 'res4b_branch2c': 0.1,
            # 'res5c_branch2b': 0.3,
            # 'res5c_branch2c': 0.1,
            'add_11': 0.2,
            # 'add_16': 0.3,
        }
        self.model = resnet50.ResNet50(weights='imagenet', include_top=False)
        print(self.model.summary()) # show layer_contribitions

    def mobilenet_v2_model(self):
        self.model_name = 'mobilenet_v2'
        self.layer_contributions = {
            # 'block_1_expand': 0.3,
            # 'block_4_add': 0.2,
            # 'block_9_add': 0.2,
            # 'block_11_project': 0.1,
            # 'block_15_add': 0.5,
            'Conv_1': 0.2,
        }
        self.model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
        print(self.model.summary()) # show layer_contribitions
