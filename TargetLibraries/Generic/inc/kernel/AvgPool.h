/* =====================================================================
 * Title:        AvgPool.h
 * Description:  Header for Average Pooling operations
 *
 * Date:         29.04.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
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

#ifndef __DEEPLOY_BASIC_MATH_POOLING_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_POOLING_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements avg pooling operations.
 *
 * The input is a multi-dimensional tensor in NCHW format,
 * the output is the pooled tensor in NCHW format.
 */

/******************************************************************************/
/*                      Average Pooling (8bit)                                */
/******************************************************************************/

/*
 * 2D Average Pooling  ----------------------------------
 * kernel      = AvgPool2d_s8_s8_NCHW
 * layout      = NCHW
 * data type   = 8-bit integer
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 */
void AvgPool2d_s8_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                          uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                          uint32_t SP, uint32_t SQ, uint32_t pool_size,
                          int8_t *__restrict__ pDstC, int32_t input_offset,
                          int32_t output_offset);

/*
 * 2D Average Pooling (Float32) ----------------------------------
 * kernel      = AvgPool2d_fp32_fp32_NCHW
 * layout      = NCHW
 * data type   = 32-bit float
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 */
void AvgPool2d_fp32_fp32_NCHW(float const *__restrict__ pSrcA, uint32_t C,
                              uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                              uint32_t SP, uint32_t SQ, uint32_t pool_size,
                              float *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_POOLING_KERNEL_HEADER_