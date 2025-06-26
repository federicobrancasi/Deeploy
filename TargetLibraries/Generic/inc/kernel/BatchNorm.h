/* =====================================================================
 * Title:        BatchNorm.h
 * Description:  Header for Batch Normalization operations
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

#ifndef __DEEPLOY_BASIC_MATH_BATCHNORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_BATCHNORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements batch normalization operations.
 *
 * The input is a multi-dimensional tensor in NCHW format,
 * the output is the normalized tensor in NCHW format.
 */

/******************************************************************************/
/*                      Batch Normalization (8bit)                            */
/******************************************************************************/

/*
 * 2D Batch Normalization  ----------------------------------
 * kernel      = BatchNorm2d_s8_s8_NCHW
 * layout      = NCHW
 * data type   = 8-bit integer
 * simd        = no
 */
void BatchNorm2d_s8_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                            uint32_t H, uint32_t W,
                            int8_t const *__restrict__ weight,
                            int8_t const *__restrict__ bias,
                            int32_t const *__restrict__ running_mean,
                            int32_t const *__restrict__ running_var,
                            float32_t eps, int8_t *__restrict__ pDstC);

/*
 * 2D Batch Normalization (Float32) ----------------------------------
 * kernel      = BatchNorm2d_fp32_fp32_NCHW
 * layout      = NCHW
 * data type   = 32-bit float
 * simd        = no
 */
void BatchNorm2d_fp32_fp32_NCHW(float32_t const *__restrict__ pSrcA, uint32_t C,
                                uint32_t H, uint32_t W,
                                float32_t const *__restrict__ weight,
                                float32_t const *__restrict__ bias,
                                float32_t const *__restrict__ running_mean,
                                float32_t const *__restrict__ running_var,
                                float32_t eps, float32_t *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_BATCHNORM_KERNEL_HEADER_