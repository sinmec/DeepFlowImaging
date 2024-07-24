from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model

from conv2d_block import conv2d_block


def create_mini_model(
        window_size, n_filters, dropouts, kernel_sizes, max_pool_size, batchnorm, **kwargs
):
    img_size = (window_size, window_size)

    input_img = Input((img_size[0], img_size[1], 1), name="img")

    # Contracting Path
    c1 = conv2d_block(
        input_img, n_filters[0], kernel_size=kernel_sizes[0], batchnorm=batchnorm
    )
    p1 = MaxPooling2D((max_pool_size, max_pool_size))(c1)
    p1 = Dropout(dropouts[0])(p1)

    c2 = conv2d_block(
        p1, n_filters[1], kernel_size=kernel_sizes[1], batchnorm=batchnorm
    )
    p2 = MaxPooling2D(max_pool_size, max_pool_size)(c2)
    p2 = Dropout(dropouts[1])(p2)

    u4 = Conv2DTranspose(
        n_filters[2],
        kernel_size=kernel_sizes[2],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c2)
    u4 = concatenate([u4, c1])
    u4 = Dropout(dropouts[1])(u4)
    c4 = conv2d_block(
        u4, n_filters[3], kernel_size=kernel_sizes[3], batchnorm=batchnorm
    )

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c4)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def create_econ_model(
        window_size, n_filters, dropouts, kernel_sizes, max_pool_size, batchnorm, **kwargs
):
    i = 0
    j = 0
    k = 0
    img_size = (window_size, window_size)
    input_img = Input((img_size[0], img_size[1], 1), name="img")

    # Contracting Path
    c1 = conv2d_block(
        input_img, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p1 = MaxPooling2D((max_pool_size, max_pool_size))(c1)
    p1 = Dropout(dropouts[j])(p1)
    i += 1
    k += 1
    j += 1

    c2 = conv2d_block(
        p1, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p2 = MaxPooling2D((max_pool_size, max_pool_size))(c2)
    p2 = Dropout(dropouts[j])(p2)
    i += 1
    k += 1
    j += 1

    c3 = conv2d_block(
        p2, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1

    u4 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c3)
    i += 1
    k += 1
    u4 = concatenate([u4, c2])
    u4 = Dropout(dropouts[j])(u4)
    c4 = conv2d_block(
        u4, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    u5 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c4)
    i += 1
    k += 1
    u5 = concatenate([u5, c1])
    u5 = Dropout(dropouts[j])(u5)
    c5 = conv2d_block(
        u5, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def create_large_model(
        window_size, n_filters, dropouts, kernel_sizes, max_pool_size, batchnorm, **kwargs
):
    i = 0
    j = 0
    k = 0
    img_size = (window_size, window_size)
    input_img = Input((img_size[0], img_size[1], 1), name="img")

    # Contracting Path
    c1 = conv2d_block(
        input_img, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p1 = MaxPooling2D((max_pool_size, max_pool_size))(c1)
    p1 = Dropout(dropouts[j])(p1)
    i += 1
    k += 1
    j += 1

    c2 = conv2d_block(
        p1, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p2 = MaxPooling2D((max_pool_size, max_pool_size))(c2)
    p2 = Dropout(dropouts[j])(p2)
    i += 1
    k += 1
    j += 1

    c3 = conv2d_block(
        p2, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p3 = MaxPooling2D((max_pool_size, max_pool_size))(c3)
    p3 = Dropout(dropouts[j])(p3)
    i += 1
    k += 1
    j += 1

    c4 = conv2d_block(
        p3, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1

    # Expansive Path
    u5 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c4)
    i += 1
    k += 1
    u5 = concatenate([u5, c3])
    u5 = Dropout(dropouts[j])(u5)
    c5 = conv2d_block(
        u5, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    u6 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c5)
    i += 1
    k += 1
    u6 = concatenate([u6, c2])
    u6 = Dropout(dropouts[j])(u6)
    c6 = conv2d_block(
        u6, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    u7 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c6)
    i += 1
    k += 1
    u7 = concatenate([u7, c1])
    u7 = Dropout(dropouts[j])(u7)
    c7 = conv2d_block(
        u7, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c7)

    model = Model(inputs=[input_img], outputs=[outputs])

    # plot_model(
    #     model, to_file="debug_UNET_large.png", show_shapes=True, show_layer_names=True
    # )

    return model


def create_largest_model(
        window_size, n_filters, dropouts, kernel_sizes, max_pool_size, batchnorm, **kwargs
):
    i = 0
    j = 0
    k = 0
    img_size = (window_size, window_size)
    input_img = Input((img_size[0], img_size[1], 1), name="img")

    # Contracting Path
    c1 = conv2d_block(
        input_img, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p1 = MaxPooling2D((max_pool_size, max_pool_size))(c1)
    p1 = Dropout(dropouts[j])(p1)
    i += 1
    k += 1
    j += 1

    c2 = conv2d_block(
        p1, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p2 = MaxPooling2D((max_pool_size, max_pool_size))(c2)
    p2 = Dropout(dropouts[j])(p2)
    i += 1
    k += 1
    j += 1

    c3 = conv2d_block(
        p2, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p3 = MaxPooling2D((max_pool_size, max_pool_size))(c3)
    p3 = Dropout(dropouts[j])(p3)
    i += 1
    k += 1
    j += 1

    c4 = conv2d_block(
        p3, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    p4 = MaxPooling2D((max_pool_size, max_pool_size))(c4)
    p4 = Dropout(dropouts[j])(p4)
    i += 1
    k += 1
    j += 1

    c5 = conv2d_block(
        p4, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1

    # Expansive Path
    u6 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c5)
    i += 1
    k += 1
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropouts[j])(u6)
    c6 = conv2d_block(
        u6, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    u7 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c6)
    i += 1
    k += 1
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropouts[j])(u7)
    c7 = conv2d_block(
        u7, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    u8 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c7)
    i += 1
    k += 1
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropouts[j])(u8)
    c8 = conv2d_block(
        u8, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    u9 = Conv2DTranspose(
        n_filters[i],
        kernel_size=kernel_sizes[k],
        strides=(max_pool_size, max_pool_size),
        padding="same",
    )(c8)
    i += 1
    k += 1
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropouts[j])(u9)
    c9 = conv2d_block(
        u9, n_filters[i], kernel_size=kernel_sizes[k], batchnorm=batchnorm
    )
    i += 1
    k += 1
    j += 1

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    # plot_model(
    #     model, to_file="debug_UNET_largest.png", show_shapes=True, show_layer_names=True
    # )

    return model
