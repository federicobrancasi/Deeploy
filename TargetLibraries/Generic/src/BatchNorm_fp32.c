/* =====================================================================
 * Title:        BatchNorm_fp32.c
 * Description:  Float32 version of BatchNorm2D
 *
 * Date:         30.04.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Federico Brancasi, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"
#include <math.h>

void BatchNorm2d_fp32_fp32_NCHW(float32_t const *__restrict__ pSrcA, uint32_t C,
                                uint32_t H, uint32_t W,
                                float32_t const *__restrict__ weight,
                                float32_t const *__restrict__ bias,
                                float32_t const *__restrict__ running_mean,
                                float32_t const *__restrict__ running_var,
                                float32_t eps,
                                float32_t *__restrict__ pDstC)
{
    for (uint32_t c = 0; c < C; c++)
    {
        float32_t mean = running_mean[c];
        float32_t std_dev = sqrtf(running_var[c] + eps);
        float32_t gamma = weight[c];
        float32_t beta = bias[c];

        for (uint32_t h = 0; h < H; h++)
        {
            for (uint32_t w = 0; w < W; w++)
            {
                uint32_t idx = c * H * W + h * W + w;
                float32_t centered = pSrcA[idx] - mean;
                float32_t normalized = centered / std_dev;
                float32_t scaled = normalized * gamma;
                pDstC[idx] = scaled + beta;
            }
        }
    }
}