// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/reduction.h"

namespace phi {
namespace distributed {

SpmdInfo LogSumExpInferSpmd(const DistMetaTensor& x,
                            const std::vector<int>& axis,
                            bool keepdims,
                            bool reduce_all);

SpmdInfo LogSumExpInferSpmdReverse(const DistMetaTensor& x,
                                   const DistMetaTensor& out,
                                   const std::vector<int>& axis,
                                   bool keepdims,
                                   bool reduce_all);

SpmdInfo LogSumExpGradInferSpmd(const DistMetaTensor& x,
                                const DistMetaTensor& out,
                                const DistMetaTensor& out_grad,
                                const std::vector<int>& axis,
                                bool keepdims,
                                bool reduce_all);
}  // namespace distributed
}  // namespace phi