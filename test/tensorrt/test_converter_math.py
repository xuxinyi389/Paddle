# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestMaxTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.max
        self.api_args = {
            "x": np.random.randn(2, 4).astype(np.float32),
            "axis": [0, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestDivideTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.divide
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(2, 3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMultiplyTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.multiply
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(2, 3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSubstractTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.subtract
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(2, 3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAddTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.add
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.randn(2, 3).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRemainderFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.remainder
        self.api_args = {
            "x": np.random.randn(2, 3).astype(np.float32),
            "y": np.random.uniform(low=0.1, high=1, size=(2, 3)).astype(
                np.float32
            ),  # Ensure y is non-zero
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randn(*shape).astype(np.float32),
            "y": lambda shape: np.random.uniform(
                low=0.1, high=1, size=shape
            ).astype(np.float32),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestRemainderIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.remainder
        self.api_args = {
            "x": np.random.randint(1, 10, size=(2, 3)).astype(np.int32),
            "y": np.random.randint(1, 10, size=(2, 3)).astype(
                np.int32
            ),  # Ensure y is non-zero
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(1, 10, size=shape).astype(
                np.int32
            ),
            "y": lambda shape: np.random.randint(1, 10, size=shape).astype(
                np.int32
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMinTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.min
        self.api_args = {
            "x": np.random.randn(2, 4).astype(np.float32),
            "axis": [0, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSumTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.api_args = {
            "x": np.random.randn(2, 4, 6).astype(np.int32),
            "axis": [1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 6]}
        self.max_shape = {"x": [5, 4, 6]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSum1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sum
        self.api_args = {
            "x": np.random.randn(2, 4, 6).astype(np.float32),
            "axis": [1, 1],
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4, 6]}
        self.max_shape = {"x": [5, 4, 6]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAnyTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.any
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype(np.bool_),
            "axis": [1, 1],
            "keepdim": True,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAny1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.any
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype(np.bool_),
            "axis": [1, 1],
            "keepdim": False,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAllTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.all
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype(np.bool_),
            "axis": [1, 1],
            "keepdim": True,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAll1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.all
        self.api_args = {
            "x": np.random.randn(2, 3, 2).astype(np.bool_),
            "axis": [1, 1],
            "keepdim": False,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 2]}
        self.max_shape = {"x": [5, 3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFloorDivideFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.floor_divide
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFloorDivideIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.floor_divide
        self.api_args = {
            "x": np.random.randint(low=1, high=100, size=(2, 3), dtype="int64"),
            "y": np.random.randint(low=1, high=100, size=(2, 3), dtype="int64"),
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
            "y": lambda shape: np.random.randint(
                1, 100, size=shape, dtype="int64"
            ),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.log
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.log
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
