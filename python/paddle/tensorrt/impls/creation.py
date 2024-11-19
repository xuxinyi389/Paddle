# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorrt as trt

import paddle
from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    cast_tensor,
    trt_cast,
    trt_floor_div,
    trt_max,
    trt_reduce_to_scalar,
    trt_reshape,
    trt_sub,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.full_int_array", trt_version="8.x")
def full_int_array_converter(network, paddle_op, inputs):
    value = paddle_op.attrs()["value"]
    if len(value) == 0:
        return ()
    value_weight = trt.Weights(np.array(value, dtype=np.int32))
    full_int_array_layer = network.add_constant([len(value)], value_weight)
    return full_int_array_layer.get_output(0)


@converter_registry.register("pd_op.full", trt_version="8.x")
def full_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["shape"]
    value = paddle_op.attrs().get("value", 1.0)
    full_layer = network.add_constant(
        shape, np.full(shape, value, dtype=np.float32)
    )
    return full_layer.get_output(0)


@converter_registry.register("pd_op.assign", trt_version="8.x")
@converter_registry.register("pd_op.assign_out_", trt_version="8.x")
def assign_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    identity_layer = network.add_identity(input_tensor)
    return identity_layer.get_output(0)


@converter_registry.register("pd_op.assign_value_", trt_version="8.x")
def assign_value_converter(network, paddle_op, inputs):
    attrs = paddle_op.attrs()
    shape = attrs['shape']
    dtype = attrs['dtype']
    values = attrs['values']

    if dtype == paddle.float32:
        dtype = np.float32

    constant_layer = network.add_constant(shape, np.array(values, dtype=dtype))
    return constant_layer.get_output(0)


@converter_registry.register("pd_op.arange", trt_version="8.x")
def arange_converter(network, paddle_op, inputs):
    start, end, step = inputs
    zero_tensor = add_1D_constant_layer(network, 0, np.int32)

    delta = trt_sub(network, end, start)

    f_quotient_tensor = trt_floor_div(network, delta, step)

    if start.dtype == trt.DataType.FLOAT:
        quotient_tensor = cast_tensor(network, f_quotient_tensor, trt.int32)
    else:
        quotient_tensor = f_quotient_tensor

    number_tensor = trt_max(network, quotient_tensor, zero_tensor)

    reshape_start_layer = trt_reshape(network, start, (1,))

    start_tensor = trt_reduce_to_scalar(network, reshape_start_layer)

    fill_layer = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
    fill_layer.set_input(0, number_tensor)
    fill_layer.set_input(1, start_tensor)
    fill_layer.set_input(2, step)

    return fill_layer.get_output(0)


@converter_registry.register("pd_op.full_like", trt_version="8.x")
def full_like_converter(network, paddle_op, inputs):
    shape = tuple(paddle_op.operands()[0].source().shape)
    ndims = len(shape)

    out_dtype = int(paddle_op.attrs().get("dtype", None))
    # Reference paddle/phi/common/data_type.h enum DataType
    if out_dtype == 1:  # paddle.bool
        out_dtype = trt.int32
    elif out_dtype == 7:  # paddle.int32
        out_dtype = trt.int32
    elif out_dtype == 9:  # paddle.int64
        out_dtype = trt.int32
    elif out_dtype == 10:  # paddle.float32
        out_dtype = trt.float32
    elif out_dtype == 11:  # paddle.float64
        out_dtype = trt.float32
    else:
        raise RuntimeError(
            f"cast converter currently doesn't support dtype: {out_dtype}"
        )

    value_op = paddle_op.operands()[1].source().get_defining_op()
    if value_op.name() == "pd_op.full":
        fill_value = value_op.attrs()["value"]
        value = network.add_constant(
            (1,),
            np.array(
                [
                    fill_value,
                ],
                dtype=np.float32,
            ),
        ).get_output(0)
        value = trt_cast(network, value, out_dtype)
    else:
        value = inputs[1]

    shuffle_layer = network.add_shuffle(value)
    shuffle_layer.reshape_dims = (1,) * ndims

    start_vec = np.zeros((ndims,), dtype=np.int32)
    start_tensor = network.add_constant((ndims,), start_vec).get_output(0)
    shape_tensor = network.add_shape(inputs[0]).get_output(0)
    stride_tensor = network.add_constant(
        (ndims,), np.ones((ndims,), dtype=np.int32)
    ).get_output(0)

    slice_layer = network.add_slice(
        shuffle_layer.get_output(0),
        start_vec,
        [1] * ndims,
        np.ones((ndims,), dtype=np.int32),
    )
    slice_layer.mode = trt.SliceMode.FILL
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, shape_tensor)
    slice_layer.set_input(3, stride_tensor)
    value = trt_cast(network, value, out_dtype)
    slice_layer.set_input(4, value)
    return slice_layer.get_output(0)


@converter_registry.register("pd_op.full_with_tensor", trt_version="8.x")
def full_with_tensor_converter(network, paddle_op, inputs):
    value_input = inputs[0]

    shape_tensor = None
    dtype = paddle_op.attrs()["dtype"]

    operands = paddle_op.operands()
    num_operands = len(operands)

    if num_operands >= 2:
        shape_tensor = inputs[1]
        if isinstance(shape_tensor, list):
            shape_tensor_list = shape_tensor
        else:
            shape_tensor_list = [shape_tensor]

    shape_op = paddle_op.operands()[1].source().get_defining_op()
    if shape_op.name() == "pd_op.full_int_array":
        shape_tensor = shape_op.attrs()["value"]
        is_static_shape = True
    else:
        shape_tensor = inputs[1]
        is_static_shape = False

    shape_nbDims = 0
    tensor_rank = 0
    if isinstance(shape_tensor, trt.ITensor):
        shape_x = shape_tensor.shape
        shape_nbDims = len(shape_x)
        shapes_tensor = shape_tensor
    elif isinstance(shape_tensor, (list, tuple)):
        shape_nbDims = len(shape_tensor)
        shapes_tensor = shape_tensor
    else:
        raise TypeError(f"Unsupported shape_tensor type: {type(shape_tensor)}")

    if shape_tensor is not None and len(shape_tensor_list) == 1:
        is_dynamic_shape = True
    elif len(shape_tensor_list) >= 1:
        is_dynamic_shape = True
    else:
        is_dynamic_shape = False

    if is_dynamic_shape:
        if len(shape_tensor_list) == 1:
            shape_tensor = shape_tensor_list[0]
            if not isinstance(shape_tensor, trt.ITensor):
                raise TypeError("shape_tensor must be an ITensor")
            if len(shape_tensor.shape) != 1:
                raise ValueError("The rank of shape_tensor must be 1")
            tensor_rank = shape_tensor.shape[0]
            shapes_tensor = shape_tensor
        else:
            shape_tensors = []
            for tensor in shape_tensor_list:
                if len(tensor.shape) == 0:
                    tensor = trt_reshape(network, tensor, (1,))
                shape_tensors.append(tensor)

            concat_layer = network.add_concatenation(shape_tensors)
            shapes_tensor = concat_layer.get_output(0)
            tensor_rank = len(shape_tensors)

        fill_layer = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
        fill_layer.set_input(0, shapes_tensor)

    if dtype == paddle.int32 or dtype == paddle.int64:
        beta_vec = [0] * tensor_rank
        value_input = trt_reduce_to_scalar(network, value_input)
        fill_layer.set_input(1, value_input)
        fill_layer.set_input(
            2, add_1D_constant_layer(network, beta_vec, np.int32)
        )
    elif dtype == paddle.float32:
        beta_vec = [0.0] * tensor_rank
        value_input = trt_reduce_to_scalar(network, value_input)
        fill_layer.set_input(1, value_input)
        fill_layer.set_input(
            2, add_1D_constant_layer(network, beta_vec, np.float32)
        )
    else:
        raise ValueError(f"Unsupported dtype for full_with_tensor: {dtype}")

    output_tensor = fill_layer.get_output(0)
    return output_tensor
