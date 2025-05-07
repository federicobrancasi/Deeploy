/* =====================================================================
 * Title:        BatchNorm_s8.c
 * Description:  Int8 version of BatchNorm2D
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

void BatchNorm2d_s8_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                            uint32_t H, uint32_t W,
                            int8_t const *__restrict__ weight,
                            int8_t const *__restrict__ bias,
                            int32_t const *__restrict__ running_mean,
                            int32_t const *__restrict__ running_var,
                            float32_t eps,
                            int8_t *__restrict__ pDstC)
{
    for (uint32_t c = 0; c < C; c++)
    {
        int32_t mean = running_mean[c];
        int32_t variance = running_var[c];
        int32_t std_dev = (int32_t)sqrtf((float)(variance + (int32_t)(eps * (1 << 16))));
        int32_t gamma = weight[c];
        int32_t beta = bias[c];

        for (uint32_t h = 0; h < H; h++)
        {
            for (uint32_t w = 0; w < W; w++)
            {
                uint32_t idx = c * H * W + h * W + w;
                int32_t input_val = pSrcA[idx];
                int32_t normalized = ((input_val - mean) << 16) / std_dev;
                int32_t scaled = (normalized * gamma >> 16) + beta;

                if (scaled > 127)
                {
                    scaled = 127;
                }
                else if (scaled < -128)
                {
                    scaled = -128;
                }

                pDstC[idx] = (int8_t)scaled;
            }
        }
    }
}
