/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo SliceInferSpmd(const DistMetaTensor& input,
                        const std::vector<int64_t>& axes,
                        const std::vector<int>& starts,
                        const std::vector<int>& ends,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis);

SpmdInfo SliceInferSpmdReverse(const DistMetaTensor& input,
                               const DistMetaTensor& output,
                               const std::vector<int64_t>& axes,
                               const std::vector<int>& starts,
                               const std::vector<int>& ends,
                               const std::vector<int64_t>& infer_flags,
                               const std::vector<int64_t>& decrease_axis);

SpmdInfo SliceInferSpmdDynamic(const DistMetaTensor& input,
                               const std::vector<int64_t>& axes,
                               const IntArray& starts,
                               const IntArray& ends,
                               const std::vector<int64_t>& infer_flags,
                               const std::vector<int64_t>& decrease_axis);

SpmdInfo ViewSliceInferSpmd(const DistMetaTensor& input,
                            int64_t begin_idx,
                            int64_t end_idx);

SpmdInfo SliceGradInferSpmdDynamic(const DistMetaTensor& input,
                                   const DistMetaTensor& out_grad,
                                   const std::vector<int64_t>& axes,
                                   const IntArray& starts,
                                   const IntArray& ends,
                                   const std::vector<int64_t>& infer_flags,
                                   const std::vector<int64_t>& decrease_axis);

SpmdInfo StridedSliceInferSpmd(const DistMetaTensor& input,
                               const std::vector<int>& axes,
                               const std::vector<int>& starts,
                               const std::vector<int>& ends,
                               const std::vector<int>& strides);

SpmdInfo StridedSliceGradInferSpmd(const DistMetaTensor& input,
                                   const DistMetaTensor& out_grad,
                                   const std::vector<int>& axes,
                                   const std::vector<int>& starts,
                                   const std::vector<int>& ends,
                                   const std::vector<int>& strides);

SpmdInfo StridedSliceInferSpmdDynamic(const DistMetaTensor& input,
                                      const std::vector<int>& axes,
                                      const IntArray& starts,
                                      const IntArray& ends,
                                      const IntArray& strides);

SpmdInfo StridedSliceGradInferSpmdDynamic(const DistMetaTensor& input,
                                          const DistMetaTensor& out_grad,
                                          const std::vector<int>& axes,
                                          const IntArray& starts,
                                          const IntArray& ends,
                                          const IntArray& strides);

}  // namespace distributed
}  // namespace phi
