"""
Alternative implementation of model construction and training ops for a TensorFlow-based protein model.

This script defines how to build:
1. An inference (forward) graph from a network YAML specification.
2. A training graph that attaches loss functions, optimizers, and metrics.

The logic is equivalent to the original build_tf_model.py but is restructured
and renamed to avoid direct conflict or duplication.
"""

import argparse
import yaml
import tensorflow as tf
from os.path import isfile

# Import any custom layers/graph utilities that exist in the project.
# Make sure these references match their actual file names and function signatures in your project.
import gen_structure_graph as structure_builder
from my_pipgcn import node_average_gc  # Or rename if needed.

def parse_layer_definition(layer_def, scope_vars):
    """
    Recursively resolves or evaluates items in a layer definition that start with '~'.
    This enables referencing Python expressions, functions, etc.
    """
    if isinstance(layer_def, dict):
        return {k: parse_layer_definition(v, scope_vars) for k, v in layer_def.items()}
    elif isinstance(layer_def, list):
        return [parse_layer_definition(item, scope_vars) for item in layer_def]
    elif isinstance(layer_def, str) and layer_def.startswith("~"):
        return eval(layer_def[1:], globals(), scope_vars)
    else:
        return layer_def


def create_input_placeholders(input_shape):
    """
    Defines placeholders for raw sequence data, labels (scores), and a boolean for training control.

    :param input_shape: Tuple (batch_size, sequence_length, feature_dim).
    :return: Dictionary of placeholders.
    """
    adjusted_shape = [None] + list(input_shape[1:])  # Remove or ignore batch dimension
    placeholders = {
        "sequences": tf.compat.v1.placeholder(
            tf.float32, shape=adjusted_shape, name="placeholder_sequences"
        ),
        "labels": tf.compat.v1.placeholder(
            tf.float32, shape=None, name="placeholder_labels"
        ),
        "training_flag": tf.compat.v1.placeholder_with_default(
            False, shape=(), name="placeholder_training_flag"
        ),
    }
    return placeholders


def assemble_inference_ops(network_config_file, adjacency=None, placeholders=None):
    """
    Reads a YAML network specification and sequentially builds each layer.
    The final layer is assumed to produce a single-value output for each example.

    :param network_config_file: Path to the YAML spec describing model layers.
    :param adjacency: Optional adjacency matrix tensor (for graph convolution layers).
    :param placeholders: Dictionary of input placeholders from `create_input_placeholders`.
    :return: Tensor for model predictions.
    """
    with open(network_config_file, "r") as f:
        yaml_content = yaml.safe_load(f)

    # Start building from the raw input placeholder
    layer_outputs = [placeholders["sequences"]]

    for layer_info in yaml_content["network"]:
        # Evaluate any expressions that start with '~'
        resolved_spec = parse_layer_definition(layer_info, locals())

        # Some layers (like graph conv) require adjacency info
        if resolved_spec["layer_func"] == node_average_gc and adjacency is None:
            raise ValueError("Graph convolution layer requires an adjacency matrix.")

        # layer_func is typically a Python function object
        layer_function = resolved_spec["layer_func"]
        kwargs_for_layer = resolved_spec["arguments"]
        # Construct the layer and append to the outputs list
        current_output = layer_function(*layer_outputs[-1:], **kwargs_for_layer)
        layer_outputs.append(current_output)

    # Final output layer: a single “task” dimension
    final_output = tf.layers.dense(layer_outputs[-1], units=1, activation=None, name="final_output")
    # Squeeze to remove extra dimension
    predictions = tf.squeeze(final_output, axis=1)
    return predictions


def define_loss_operations(param_dict, pred_dict):
    """
    Creates a mean squared error loss operation for training.

    :param param_dict: Dictionary of arguments, should contain placeholders and hyperparams.
    :param pred_dict: Dictionary with predictions and placeholders.
    :return: A scalar loss tensor.
    """
    true_scores = pred_dict["placeholders"]["labels"]
    predicted_scores = pred_dict["predictions"]

    # Define MSE loss
    mse_loss = tf.compat.v1.losses.mean_squared_error(
        labels=true_scores,
        predictions=predicted_scores,
        reduction=tf.compat.v1.losses.Reduction.MEAN
    )
    return mse_loss


def configure_optimizer(param_dict, loss_tensor):
    """
    Sets up the optimizer (Adam) and global step. Minimizes the loss.

    :param param_dict: Dictionary with hyperparameters, e.g., learning rate.
    :param loss_tensor: Scalar loss to minimize.
    :return: A tuple of (global_step, train_op).
    """
    learning_rate = param_dict.get("learning_rate", 1e-3)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    global_step_var = tf.Variable(0, trainable=False, name="global_step")
    train_step = optimizer.minimize(loss_tensor, global_step=global_step_var)

    return global_step_var, train_step


def prepare_metric_placeholders():
    """
    Creates placeholders for metrics that are computed outside of TensorFlow (like Pearson r, R^2, etc.),
    so we can log them in TensorBoard.

    :return: (dict_of_metric_placeholders, placeholder_for_val_loss, placeholder_for_train_loss)
    """
    mse_ph = tf.compat.v1.placeholder(tf.float32, name="placeholder_mse")
    pearson_ph = tf.compat.v1.placeholder(tf.float32, name="placeholder_pearson")
    r2_ph = tf.compat.v1.placeholder(tf.float32, name="placeholder_r2")
    spearman_ph = tf.compat.v1.placeholder(tf.float32, name="placeholder_spearman")

    valid_loss_ph = tf.compat.v1.placeholder(tf.float32, name="placeholder_val_loss")
    train_loss_ph = tf.compat.v1.placeholder(tf.float32, name="placeholder_train_loss")

    metrics_placeholder_dict = {
        "mse": mse_ph,
        "pearsonr": pearson_ph,
        "r2": r2_ph,
        "spearmanr": spearman_ph,
    }

    return metrics_placeholder_dict, valid_loss_ph, train_loss_ph


def create_summary_ops(metric_phs, val_loss_ph, train_loss_ph):
    """
    Registers summary scalars for TensorBoard logging.

    :param metric_phs: Dictionary of placeholders for MSE, Pearson r, etc.
    :param val_loss_ph: Placeholder for validation loss.
    :param train_loss_ph: Placeholder for training loss.
    :return: Two merged summary ops: one for per-epoch logs, one for metrics logs.
    """
    tf.compat.v1.summary.scalar("validation_loss", val_loss_ph, collections=["epoch_summaries"])
    tf.compat.v1.summary.scalar("training_loss", train_loss_ph, collections=["epoch_summaries"])

    tf.compat.v1.summary.scalar("mse", metric_phs["mse"], collections=["metric_summaries"])
    tf.compat.v1.summary.scalar("pearsonr", metric_phs["pearsonr"], collections=["metric_summaries"])
    tf.compat.v1.summary.scalar("r2", metric_phs["r2"], collections=["metric_summaries"])
    tf.compat.v1.summary.scalar("spearmanr", metric_phs["spearmanr"], collections=["metric_summaries"])

    epoch_summaries_merged = tf.compat.v1.summary.merge_all(key="epoch_summaries")
    metric_summaries_merged = tf.compat.v1.summary.merge_all(key="metric_summaries")

    return epoch_summaries_merged, metric_summaries_merged


def construct_forward_graph(build_args, data_shape):
    """
    Builds the forward (inference) portion of the graph:
    1) Loads adjacency if needed.
    2) Creates placeholders for input data.
    3) Reads YAML to build layers.
    4) Outputs predictions.

    :param build_args: Dictionary with keys like 'net_file', 'graph_fn', etc.
    :param data_shape: Tuple (N, sequence_length, feature_dim). N is unused here but required for shape consistency.
    :return: A dictionary containing placeholders and the predictions tensor.
    """
    # Potentially load adjacency matrix for graph convolution layers
    adjacency_tensor = None
    if isfile(build_args["graph_fn"]):
        graph_struct = structure_builder.load_graph(build_args["graph_fn"])
        adjacency_matrix = structure_builder.ordered_adjacency_matrix(graph_struct)
        adjacency_tensor = tf.convert_to_tensor(adjacency_matrix, dtype=tf.float32)

    placeholders = create_input_placeholders(data_shape)
    predictions = assemble_inference_ops(
        network_config_file=build_args["net_file"],
        adjacency=adjacency_tensor,
        placeholders=placeholders
    )

    return {
        "placeholders": placeholders,
        "predictions": predictions
    }


def construct_training_graph(build_args, inference_dict):
    """
    Adds loss, optimizer, global step, and summary placeholders/ops to the existing inference graph.

    :param build_args: Dictionary with training parameters (learning rate, etc.).
    :param inference_dict: Dictionary from construct_forward_graph with placeholders and predictions.
    :return: A dictionary of training-related ops and placeholders.
    """
    # Prepare placeholders for external metrics and loss logging
    metric_phs, val_loss_ph, train_loss_ph = prepare_metric_placeholders()
    epoch_summaries, metric_summaries = create_summary_ops(metric_phs, val_loss_ph, train_loss_ph)

    # Build MSE loss
    loss_op = define_loss_operations(build_args, inference_dict)

    # Set up optimization
    global_step, train_op = configure_optimizer(build_args, loss_op)

    # Global initializer
    init_op = tf.compat.v1.global_variables_initializer()

    return {
        "loss": loss_op,
        "train_op": train_op,
        "global_step": global_step,
        "init_op": init_op,
        "epoch_summaries": epoch_summaries,
        "metric_summaries": metric_summaries,
        "val_loss_placeholder": val_loss_ph,
        "train_loss_placeholder": train_loss_ph,
        "metric_placeholders": metric_phs
    }


def build_entire_graph(params, encoded_shape, reset_graph=True):
    """
    High-level function that resets the graph (optionally), then constructs inference and training ops.

    :param params: Dictionary (or argparse.Namespace) containing fields:
                   {
                     'net_file': <path to YAML>,
                     'graph_fn': <path to adjacency file>,
                     'learning_rate': <float>,
                     ...
                   }
    :param encoded_shape: Shape of the encoded data (N, L, F) for placeholders.
    :param reset_graph: If True, clears the current default graph before building.
    :return: (inference_dict, training_dict)
    """
    # Convert Namespace to dict if needed
    if isinstance(params, argparse.Namespace):
        params = vars(params)

    if reset_graph:
        tf.compat.v1.reset_default_graph()

    inference_part = construct_forward_graph(params, encoded_shape)
    training_part = construct_training_graph(params, inference_part)

    return inference_part, training_part


def execute_main():
    pass


if __name__ == "__main__":
    execute_main()
