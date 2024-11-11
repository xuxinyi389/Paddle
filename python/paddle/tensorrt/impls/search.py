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


import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    squeeze_trt,
    trt_cast,
    unsqueeze_trt,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.nonzero", trt_version="8.x")
def non_zero_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    cast_layer = network.add_cast(input_tensor, trt.float32)
    non_zero_layer = network.add_non_zero(cast_layer.get_output(0))

    return non_zero_layer.get_output(0)


@converter_registry.register("pd_op.argmax", trt_version="8.x")
def argmax_converter(network, paddle_op, inputs):
    x = inputs[0]
    input_dims = x.shape
    rank = len(input_dims)
    axis = int(
        paddle_op.operands()[1]
        .source()
        .get_defining_op()
        .attrs()
        .get("value", -1)
    )
    keepdims = paddle_op.attrs()["keepdims"]

    if axis < 0:
        axis += rank

    topk_layer = network.add_topk(
        input=x, op=trt.TopKOperation.MAX, k=1, axes=(1 << axis)
    )

    if keepdims:
        return topk_layer.get_output(1)
    else:
        squeeze_layer = network.add_shuffle(topk_layer.get_output(1))
        output_dims = []
        for i in range(len(input_dims)):
            if i == axis:
                continue
            output_dims.append(input_dims[i])
        squeeze_layer.reshape_dims = tuple(output_dims)
        return squeeze_layer.get_output(0)


@converter_registry.register("pd_op.where", trt_version="8.x")
def where_converter(network, paddle_op, inputs):
    condition = inputs[0]
    x = inputs[1]
    y = inputs[2]

    select_layer = network.add_select(condition, x, y)

    return select_layer.get_output(0)


@converter_registry.register("pd_op.topk", trt_version="8.x")
def topk_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]

    input_shape = paddle_op.operands()[0].source().shape

    k = paddle_op.attrs().get("k", 1)
    axis = paddle_op.attrs().get("axis", -1)
    largest = paddle_op.attrs().get("largest", True)
    flag = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN

    input_rank = len(input_shape)

    expand_to_2d = input_rank == 1
    if expand_to_2d:
        input_tensor = unsqueeze_trt(network, input_tensor, [1])

    input_type = input_tensor.dtype
    if input_type == trt.DataType.INT32:
        input_tensor = trt_cast(network, input_tensor, trt.DataType.FLOAT)

    if axis < 0:
        axis += input_rank
    layer = network.add_topk(input_tensor, flag, k, 1 << axis)
    values = layer.get_output(0)
    indices = layer.get_output(1)

    if expand_to_2d:
        values = squeeze_trt(network, values, [1])
        indices = squeeze_trt(network, indices, [1])

    if input_type == trt.DataType.INT32:
        values = trt_cast(network, values, trt.DataType.INT32)

    return values, indices
