import keras
from keras import layers
import numpy as np
#coding=utf-8

latent_dim = 2
height = 64
width = 64
channels = 1
generator_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)#将输入转换成16*16 128通道的特征图

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.advanced_activations.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=4, padding='same')(x)#上采样64*64
x = layers.advanced_activations.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.advanced_activations.LeakyReLU()(x)

#产生64x64 1通道的特征图
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)#将(latent_dim,)->(64,64,3)
#generator.summary()



discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)#二分类

discriminator = keras.models.Model(discriminator_input, x)
#discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008,
        clipvalue=1.0,decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer,
    loss='binary_crossentropy')




discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input,gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004,clipvalue=1.0,
        decay=1e-8)
gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')




import os
from keras.preprocessing import image

from PIL import Image#python2&3

path = '/Users/apple/Desktop/IGEM/post/GAN/WormsMorphology'
x_train = []
for imagename in os.listdir(path):
    try:
        img = Image.open(path+'/'+imagename)
        data = np.array(img)
        if data.shape[0] + data.shape[1] == height + width:
            #print(data.shape)
            x_train.append(data)
        else:
            pass
    except OSError:
        continue
x_train.pop()
x_train = np.array(x_train)
print((x_train.shape[0],))
print((x_train.shape[0],)+(height,width,channels))

x_train = x_train.reshape((int(x_train.shape[0]/channels),)+(height,width,channels))
print(x_train.shape)

iterations = 10000
batch_size = 20
save_dir = '/Users/apple/Desktop/IGEM/post/GAN/WormsGenerated'#保存生成图片

start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))#正态分布随机取点
    
    generated_images = generator.predict(random_latent_vectors)#fake图
    
    stop = start + batch_size
    real_images = x_train[start: stop]
    print(generated_images.shape,real_images.shape)
    #混合真、假图片
    combined_images = np.concatenate([generated_images, real_images])
    #标签
    labels = np.concatenate([np.ones((batch_size, 1)),np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)#加随机噪声
    
    d_loss = discriminator.train_on_batch(combined_images, labels)
    
    random_latent_vectors = np.random.normal(size=(batch_size,latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
    #gan训练：训练generator，固定discriminator
    a_loss = gan.train_on_batch(random_latent_vectors,misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    if step % 100 == 0:#每100步保存一次
        gan.save_weights('gan.h5')

        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)

        img = image.array_to_img(generated_images[0] * 255., scale = False)
        img.save(os.path.join(save_dir,'GeneratedWorms' + str(step) + '.png'))
        
        #img = image.array_to_img(real_images[0] * 255., scale = False)
        #img.save(os.path.join(save_dir,'RealWorms' + str(step) + '.png'))
































