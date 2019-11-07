import os
from keras import backend as K

from functions import *
from models_config import Model


IMAGE_NAME = 'kek.jpg'
IMAGE_SAVE_PATH = 'images'
IMAGE_BASE_PATH = 'D:\images'
    

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


if __name__ == '__main__':
    K.set_learning_phase(0)

    m = Model()
    """choose model"""
    # m.inception_model()
    m.vgg16_model()
    # m.xception_model()
    # m.resnet50_model()
    # m.mobilenet_v2_model()
    
    layer_dict = dict([(layer.name, layer) for layer in m.model.layers])
    loss = K.variable(0.)
    for layer_name in m.layer_contributions:
        coeff = m.layer_contributions[layer_name]
        activation = layer_dict[layer_name].output
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        loss = loss + \
            (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)

    dream = m.model.input
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    # изменяя следующие параметры можно добиться различных эффектов
    step = 0.01 # размер шага градиентного восхождения
    num_octave = 3 # кол-во масштабов, на которых выполняется град. восхождение
    octave_scale = 1.4 # отношение между соседними масштабами (в данном случае картинка на каждом шаге будет увеилчиваться в 1.4 раза)
    iterations = 20 # 

    max_loss = 10. # Максимальная величина потерь, чем больше, тем больше безобразных эффектов может появиться на изображении

    base_image_path = '{0}\{1}'.format(IMAGE_BASE_PATH, IMAGE_NAME)

    img = preprocess_image(base_image_path)
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i))
                       for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)

        image_name_without_exp = IMAGE_NAME.split('.')[:-1][0]
        dir_save = '{0}/{1}_{2}'.format(IMAGE_SAVE_PATH, image_name_without_exp, m.model_name)
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)
        save_img(img, fname='{0}/{1}{2}{3}'.format(dir_save, 'dream_at_scale_', str(shape), '.png'))
    save_img(img, fname='{0}/{1}'.format(dir_save, 'final_dream.png'))
