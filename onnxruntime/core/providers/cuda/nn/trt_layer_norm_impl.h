/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_fp16.h"

namespace onnxruntime {
namespace cuda {

// LayerNorm kernel from layerNorm plugin of TensorRT 8.5
template <typename T>
void computeLayerNorm(int32_t const gridSize, int32_t const nHiddenDimension,
                      T const* input, T const* gamma, T const* beta,
                      T* output, float const epsilon, cudaStream_t stream);

}  // namespace cuda
}  // namespace onnxruntime
