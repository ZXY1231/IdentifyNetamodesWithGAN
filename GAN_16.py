import keras
from keras import layers
from keras import initializers
from keras import regularizers
import numpy as np
#coding=utf-8
#Windows version
# | ZhouXinyu | 2018.11.15 |Version 1.0 |...
# | ZhouXinyu | 2018.11.17 |Version 1.1 |Solving the generator problem mode collapse marker=11
latent_dim = 4
height = 64
width = 64
channels = 1

iterations = 9000
batch_size = 10
train_dis = 3

def saveGeneratedWorms(images,step):
    for worm in range(len(images)):
        img = image.array_to_img(images[worm] * 255., scale = False)
        img.save(os.path.join(save_dir,'GeneratedWorms' + str(step)+'-'+str(worm)+ '.png'))
        #img = image.array_to_img(real_images[0] * 255., scale = False)
        #img.save(os.path.join(save_dir,'RealWorms' + str(step) + '.png'))



generator_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(32 * 16 * 16)(generator_input)
x = layers.advanced_activations.LeakyReLU(alpha=0.001)(x)
x = layers.Reshape((16, 16, 32))(x)#将输入转换成16*16通道的特征图

x = layers.Conv2D(64, 5, padding='same')(x)##kernel_initializer=initializers.random_normal(mean = 1000)
x = layers.advanced_activations.LeakyReLU(alpha=0.001)(x)

x = layers.Conv2DTranspose(64, 4, strides=4, padding='same')(x)#上采样64*64#11
x = layers.advanced_activations.LeakyReLU()(x)

x = layers.Conv2D(32, 5, padding='same')(x)
x = layers.advanced_activations.LeakyReLU(alpha=0.001)(x)

#产生64x64 1通道的特征图
x = layers.Conv2D(channels, 7,activation='tanh', padding='same')(x)##activation='relu'
generator = keras.models.Model(generator_input, x)#将(latent_dim,)->(64,64,1)
#generator.summary()


discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(16, 3)(discriminator_input)
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Conv2D(32, 4)(x)#11
x = layers.advanced_activations.LeakyReLU()(x)
x = layers.Conv2D(64, 4)(x)#11
x = layers.advanced_activations.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dropout(0.1)(x)

#x = layers.Dense(1, activation='sigmoid')(x)
x = layers.Dense(1)(x)#二分类 11

discriminator = keras.models.Model(discriminator_input, x)
#discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.00008,clipvalue=1,decay=1e-8)#11
discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')


discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input,gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004,clipvalue=0.5,
        decay=1e-8)
gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')




import os
from keras.preprocessing import image
from PIL import Image#python2&3
import time

time1 = time.clock()
path = r'\\10.20.13.222\igem\WormTrack\WormTrack_summer\WormsMorphology'
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
leng = len(x_train)
print(x_train.shape)
time2 = time .clock()
print('Time',time2-time1)

save_dir = r'\\10.20.13.222\igem\WormTrack\WormTrack_summer\WormsGenerated'#保存生成图片

start = 0

gan.load_weights(r'\\10.20.13.222\igem\WormTrack\WormTrack_summer\GAN\GAN_Weight\gan.h5')#continue training from last training

for step in range(900,iterations):
    print("\n\n\nStep:",step)
    time3 = time.clock()
    #train discriminator for train_dis times
    for d_train in range(train_dis):
        random_latent_vectors = np.random.normal( size = (batch_size, latent_dim))#正态分布随机取点 ##loc = 0, scale = 1,
        #print(random_latent_vectors.shape)
        generated_images = generator.predict(random_latent_vectors)#fake图
        print()
        print(generated_images.max(),generated_images.min(),generated_images.mean())
        time4 = time.clock()
        #print('TimeGenerateFake',time4-time3)
        stop = start + batch_size
        if stop > leng :
            real_images = x_train[start:leng]+x_train[0:stop%leng]
            start = start%leng
        else:
            real_images = x_train[start: stop]
        real_images = [(i-i.min())/i.max() for i in real_images]
        real_images = np.array(real_images)
        print(real_images.max(),generated_images.min(),real_images.mean())
        #print(generated_images.shape,real_images.shape)
        time4 = time.clock()

        #mix fake images with real ones
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((batch_size, 1)),np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)#加随机噪声
        d_loss = discriminator.train_on_batch(combined_images, labels)

        start = (start + batch_size)%leng
        if start > leng - batch_size:
            start = 0

    time5 = time.clock()
    print('TimeDiscriminator',time5-time3)

    random_latent_vectors = np.random.normal( size = (batch_size,latent_dim))
    misleading_targets = np.zeros((batch_size, 1))
    #gan训练：训练generator，固定discriminator
    a_loss = gan.train_on_batch(random_latent_vectors,misleading_targets)
    time6 = time.clock()
    print('TimeGAN',time6-time5)




    if step % 100 == 0:#每100步保存一次
        gan.save_weights(r'\\10.20.13.222\igem\WormTrack\WormTrack_summer\GAN\GAN_Weight\gan.h5')

        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        #saveGeneratedWorms(generated_images,step)
        img = image.array_to_img(generated_images[0] * 255., scale = False)
        img.save(os.path.join(save_dir,'GeneratedWorms' + str(step) + '.png'))
    if step == iterations-1:
        saveGeneratedWorms(generated_images,step)
    
        #img = image.array_to_img(real_images[0] * 255., scale = False)
        #img.save(os.path.join(save_dir,'RealWorms' + str(step) + '.png'))
































