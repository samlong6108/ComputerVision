import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
import CartoontoReal.CartoontoReal
from model import get_discriminator, get_resnet_generator
from datetime import datetime
from tensorflow.keras.utils import plot_model
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--GPU",type=int, default=0)
parser.add_argument("--epoch",type=int, default=200)
parser.add_argument("--batch_size",type=int, default=2)
parser.add_argument("--buffer_size",type=int, default=100)

parser.add_argument("--lambda_cycle",type=float, default=10.0)
parser.add_argument("--lambda_identity",type=float, default=0.5)
parser.add_argument("--lambda_gamma",type=float, default=0.0)

args = parser.parse_args()



def set_tensorflow_config(per_process_gpu_memory_fraction=0.9):
    config = tf.compat.v1.ConfigProto()
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    config.gpu_options.allow_growth=True
    # sess = tf.Session(config=config)
    sess = tf.compat.v1.Session(config=config)
    
    print("== TensorFlow Config options set ==")
    print("\nThis process will now utilize {} GPU Memeory Fraction".format(per_process_gpu_memory_fraction))
set_tensorflow_config()

class LRTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):  
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
autotune = tf.data.experimental.AUTOTUNE
# Define the standard image size.
orig_img_size = (256, 256)
# Size of the random crops to be used during training.
input_img_size = (256, 256, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = args.buffer_size
batch_size = args.batch_size  

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img_r, img_c):
#     label = label+"edit"
    
    img_r = img_r
    img_c = img_c
    # Random flip
    # img_r = tf.image.random_flip_left_right(img_r)
    # img_c = tf.image.random_flip_left_right(img_c)
    # Resize to the original size first
    img_r = tf.image.resize(img_r, [*orig_img_size])
    img_c = tf.image.resize(img_c, [*orig_img_size])
    # Random crop to 256X256
    # img_r = tf.image.random_crop(img_r, size=[*input_img_size])
    # img_c = tf.image.random_crop(img_c, size=[*input_img_size])
    # Normalize the pixel values in the range [-1, 1]
    img_r = normalize_img(img_r)
    img_c = normalize_img(img_c)
    return img_r, img_c


def preprocess_test_image(img_r, img_c):
    # Only resizing and normalization for the test images.
    img_r = tf.image.resize(img_r, [input_img_size[0], input_img_size[1]])
    img_r = normalize_img(img_r)
    img_c = tf.image.resize(img_c, [input_img_size[0], input_img_size[1]])
    img_c = normalize_img(img_c)
    return img_r, img_c

datasets, _= tfds.load('CartoontoReal', download=False, with_info=True, as_supervised=True)
# train_A, train_B = datasets["trainR"], datasets["trainC"]
# test_A, test_B = datasets["testR"], datasets["testC"]
train_A = datasets["trainR"]
test_A = datasets["testR"]

train_A = (
    train_A.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)

)
# train_B = (
#     train_B.map(preprocess_train_image, num_parallel_calls=autotune)
#     .cache()
#     .shuffle(buffer_size)
#     .batch(batch_size)
# )
test_A = (
    test_A.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    #.shuffle(buffer_size)
    .batch(batch_size)
)
# test_B = (
#     test_B.map(preprocess_test_image, num_parallel_calls=autotune)
#     .cache()
#     #.shuffle(buffer_size)
#     .batch(batch_size)
# )

# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")


class CycleGan(keras.Model):
    def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        # self.lambda_cycle = lambda_cycle
        # self.lambda_identity = lambda_identity
        # self.lambda_gamma = lambda_gamma

    def compile(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer, gen_loss_fn, disc_loss_fn):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
  
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x, real_x_feature = self.disc_X(real_x, training=True) 
            disc_fake_x, fake_x_feature = self.disc_X(fake_x, training=True)
            _, cycled_x_feature = self.disc_X(cycled_x, training=True)

            disc_real_y, real_y_feature = self.disc_Y(real_y, training=True)
            disc_fake_y, fake_y_feature = self.disc_Y(fake_y, training=True)
            _, cycled_y_feature = self.disc_Y(cycled_y, training=True)

            

            # Generator adverserial loss
            # gen_G_loss = self.generator_loss_fn(disc_fake_y)
            # gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # -------------------------------------------------------------
            # Generator custom adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y, disc_real_y, fake_y, real_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x, disc_real_x, fake_x, real_x)
            # -------------------------------------------------------------

            coff1 = args.lambda_gamma
            coff2 = 1 - coff1

            # Generator cycle loss
            cycle_loss_G = (coff2 * self.cycle_loss_fn(real_y, cycled_y) + coff1 * self.cycle_loss_fn(real_y_feature, cycled_y_feature)) * args.lambda_cycle
            cycle_loss_F = (coff2 * self.cycle_loss_fn(real_x, cycled_x) + coff1 * self.cycle_loss_fn(real_x_feature, cycled_x_feature)) * args.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * args.lambda_cycle
                * args.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * args.lambda_cycle
                * args.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_A.take(self.num_img)):
            img = img[1]
            prediction = self.model.gen_F(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
            
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            # prediction.save(
            #     "picture/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            # )
        plt.savefig(f"./picture/v5/generated_img_{epoch+1}.png")
        plt.clf()
        # plt.show()
        # plt.close()
        if args.lambda_cycle > 0.6 :
            args.lambda_cycle -= 0.5
        if args.lambda_gamma < 0.9 :
            args.lambda_gamma += 0.01
        print(args.lambda_cycle)
        print(args.lambda_gamma)

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

def generator_loss_fn_custom(fake, true, fake_image, gt_image):
    image_loss_fn = keras.losses.MeanAbsoluteError()

    lambda_ = 0.5
    fake_loss = adv_loss_fn(true, fake)
    img_loss = image_loss_fn(fake_image, gt_image)
    return lambda_ * fake_loss + (1-lambda_) * img_loss
    

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_coef = random.uniform(0.9, 1.0)
    fake_coef = random.uniform(0, 0.1)

    real_loss = adv_loss_fn(real_coef * tf.ones_like(real), real)
    fake_loss = adv_loss_fn(fake_coef * tf.ones_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)


# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn_custom,
    disc_loss_fn=discriminator_loss_fn,
)
# Callbacks
plotter = GANMonitor()
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"logs/cycle_gan_v5/" + start_time
tensorboard_callback = LRTensorBoard(log_dir=logdir)

checkpoint_filepath = "./model_checkpoints/v5/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath
)


# weight_file = "./model_checkpoints/cyclegan_checkpoints.100"
# cycle_gan_model.load_weights(weight_file).expect_partial()
print("Weights loaded successfully")
plot_model(gen_G, show_shapes=True, show_layer_names=True,to_file='gen_G.png')
plot_model(gen_F, show_shapes=True, show_layer_names=True,to_file='gen_F.png')
plot_model(disc_X, show_shapes=True, show_layer_names=True,to_file='disc_X.png')
plot_model(disc_Y, show_shapes=True, show_layer_names=True,to_file='disc_Y.png')
# Here we will train the model for just one epoch as each epoch takes around
# 7 minutes on a single P100 backed machine.
cycle_gan_model.fit(
    # tf.data.Dataset.zip((train_A, train_B)),
    train_A,
    epochs=200,
    callbacks=[plotter, model_checkpoint_callback, tensorboard_callback],
    initial_epoch=0
)
