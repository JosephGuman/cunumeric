/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "cunumeric/set/unzip_indices.h"
#include "cunumeric/set/unzip_indices_template.inl"

#include <thrust/system/omp/execution_policy.h>

namespace cunumeric {

/*static*/ void UnzipIndicesTask::omp_variant(TaskContext& context)
{
  unzip_indices_template(context, thrust::omp::par);
}

}  // namespace cunumeric
