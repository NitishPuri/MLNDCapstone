
import keras.backend as K


"""
Example taken from https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
Generate and display activations produced by the given input in the intermediate layers of the model.
"""

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def save_activations(model, model_inputs):
    import numpy as np
    import matplotlib.pyplot as plt

    activation_maps = get_activations(model, model_inputs, print_shape_only = True)

    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'

        # activation_maps = activation_maps[0:5]

    for layer_num, activation_map in enumerate(activation_maps):
        layer_name = model.layers[layer_num].name
        print('Displaying activation map {}, Layer {}'.format(layer_num, layer_name))
        shape = activation_map.shape
        activation_map = activation_map[0]
        num_filters = shape[3]
        fig_size = int(np.ceil(np.sqrt(num_filters)))

        fig, ax = plt.subplots(fig_size, fig_size, figsize=(20, 20))
        fig.suptitle('{} {}'.format(layer_num, layer_name))

        if(num_filters == 1):
            img = activation_map[:,:,0]
            ax.imshow(img)
        else:        
            for i in range(num_filters):
                img = activation_map[:,:,i]
                ax[int(i/fig_size), (i%fig_size)].imshow(img)
        
        fig.savefig('img/unet_128_{}_{}.png'.format(layer_num, layer_name), dpi=fig.dpi)
        plt.close(fig)



def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        activation_map = activation_map[0]
        num_filters = shape[3]
        fig_size = int(np.ceil(np.sqrt(num_filters)))

        if(num_filters == 1):
            img = activation_map[:,:,0]
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            ax.imshow(img)
        else:        
            fig, ax = plt.subplots(fig_size, fig_size, figsize=(20, 20))
            for i in range(num_filters):
                img = activation_map[:,:,i]
                ax[int(i/fig_size), (i%fig_size)].imshow(img)

        # if len(shape) == 4:
        #     # activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        #     num_filters = shape[3]
        # elif len(shape) == 2:
        #     # try to make it square as much as possible. we can skip some activations.
        #     activations = activation_map[0]
        #     num_activations = len(activations)
        #     if num_activations > 1024:  # too hard to display it on the screen.
        #         square_param = int(np.floor(np.sqrt(num_activations)))
        #         activations = activations[0: square_param * square_param]
        #         activations = np.reshape(activations, (square_param, square_param))
        #     else:
        #         activations = np.expand_dims(activations, axis=0)
        # else:
        #     raise Exception('len(shape) = 3 has not been implemented.')
        # plt.imshow(activations, interpolation='None', cmap='jet')
        # plt.show()
