import gin

@gin.configurable
def data_shape(input=(None, None, None), output=(None)):
    image_size=(input[0], input[1])
    in_shape = input
    out_shape = output
    return image_size, in_shape, out_shape
