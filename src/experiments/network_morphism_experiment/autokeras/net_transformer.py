from copy import deepcopy
from random import randrange, sample

from experiments.network_morphism_experiment.autokeras.nn.graph import NetworkDescriptor

from experiments.network_morphism_experiment.autokeras.nn.layers import is_layer, StubDense, get_dropout_class, \
    StubReLU, get_batch_norm_class, get_pooling_class, LayerType
from experiments.network_morphism_experiment.autokeras.constant import Constant


def to_wider_graph(graph):
    weighted_layer_ids = graph.wide_layer_ids()
    weighted_layer_ids = list(filter(lambda x: graph.layer_list[x].output.shape[-1], weighted_layer_ids))
    wider_layers = sample(weighted_layer_ids, 1)

    for layer_id in wider_layers:
        layer = graph.layer_list[layer_id]
        n_add = layer.units
        graph.to_wider_model(layer_id, n_add)
    return graph


def to_skip_connection_graph(graph):
    # The last conv layer cannot be widen since wider operator cannot be done over the two sides of flatten.
    weighted_layer_ids = graph.skip_connection_layer_ids()
    valid_connection = []
    for skip_type in sorted([NetworkDescriptor.ADD_CONNECT, NetworkDescriptor.CONCAT_CONNECT]):
        for index_a in range(len(weighted_layer_ids)):
            for index_b in range(len(weighted_layer_ids))[index_a + 1:]:
                valid_connection.append((index_a, index_b, skip_type))

    if len(valid_connection) < 1:
        return graph
    for index_a, index_b, skip_type in sample(valid_connection, 1):
        a_id = weighted_layer_ids[index_a]
        b_id = weighted_layer_ids[index_b]
        if skip_type == NetworkDescriptor.ADD_CONNECT:
            graph.to_add_skip_model(a_id, b_id)
        else:
            graph.to_concat_skip_model(a_id, b_id)
    return graph


def create_new_layer(layer, n_dim):
    input_shape = layer.output.shape
    dense_deeper_classes = [StubDense, get_dropout_class(n_dim), StubReLU]
    if is_layer(layer, LayerType.RELU):
        dense_deeper_classes = [StubDense, get_dropout_class(n_dim)]
    elif is_layer(layer, LayerType.DROPOUT):
        dense_deeper_classes = [StubDense, StubReLU]

    layer_class = sample(dense_deeper_classes, 1)[0]

    if layer_class == StubDense:
        new_layer = StubDense(input_shape[0], input_shape[0])

    elif layer_class == get_dropout_class(n_dim):
        new_layer = layer_class(Constant.DENSE_DROPOUT_RATE)

    elif layer_class == get_batch_norm_class(n_dim):
        new_layer = layer_class(input_shape[-1])

    elif layer_class == get_pooling_class(n_dim):
        new_layer = layer_class(sample((1, 3, 5), 1)[0])

    else:
        new_layer = layer_class()

    return new_layer


def to_deeper_graph(graph):
    weighted_layer_ids = graph.deep_layer_ids()
    if len(weighted_layer_ids) >= Constant.MAX_LAYERS:
        return None

    deeper_layer_ids = sample(weighted_layer_ids, 1)

    for layer_id in deeper_layer_ids:
        layer = graph.layer_list[layer_id]
        new_layer = create_new_layer(layer, graph.n_dim)
        graph.to_deeper_model(layer_id, new_layer)
    return graph


def transform(graph, skip_conn=False):

    graphs = []
    for _ in range(Constant.N_NEIGHBOURS * 2):
        a = randrange(3 if skip_conn else 2)

        temp_graph = None
        if a == 0:
            temp_graph = to_deeper_graph(deepcopy(graph))
        elif a == 1:
            temp_graph = to_wider_graph(deepcopy(graph))
        elif a == 2:
            temp_graph = to_skip_connection_graph(deepcopy(graph))

        if temp_graph is not None and temp_graph.size() <= Constant.MAX_MODEL_SIZE:
            graphs.append(temp_graph)

        if len(graphs) >= Constant.N_NEIGHBOURS:
            break

    return graphs
