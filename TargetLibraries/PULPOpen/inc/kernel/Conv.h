
/* =====================================================================
 * Title:        Conv.h
 * Description:
 *
 * $Date:       05.04.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Run Wang, ETH Zurich
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

#include "DeeployPULPMath.h"

// Parameter structure for multi-core integer convolution
typedef struct {
    const int8_t *pSrcA;
    uint32_t H;
    uint32_t W;
    uint32_t C;
    const int8_t *pSrcB;
    uint32_t F;
    uint32_t P;
    uint32_t Q;
    uint32_t SP;
    uint32_t SQ;
    int32_t *pDstC;
    int32_t input_offset;
    int32_t output_offset;
    const int32_t *bias;
    uint32_t pad_top;
    uint32_t pad_bottom;
    uint32_t pad_left;
    uint32_t pad_right;
    int8_t *ctxtBuffer;
} DeeployPULPConvParams;

void PULP_Conv2d_fp32_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                    uint32_t H, uint32_t W, uint32_t C,
                                    const float32_t *__restrict__ pSrcB,
                                    uint32_t F_total, uint32_t P, uint32_t Q,
                                    uint32_t SP, uint32_t SQ,
                                    float32_t *__restrict__ pDstC,
                                    uint32_t pad_top, uint32_t pad_bottom,
                                    uint32_t pad_left, uint32_t pad_right);

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ, float32_t *__restrict__ pDstC,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, float32_t *__restrict__ pContextBuffer);

// Integer convolution functions
void Conv2d_s8_s8_s32_HWC(
    const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC, int32_t input_offset,
    int32_t output_offset, const int32_t *__restrict__ bias, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

// Multi-core optimized integer convolution functions
void PULP_Conv2d_s8_s8_s32_HWC(void *args);
void PULP_pointwise_i8_i32_i8_HWC(void *args);

// Im2Col optimized integer convolution function
void PULP_Conv2d_Im2Col_s8_s8_s32_HWC(
    const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC,
    int32_t input_offset, int32_t output_offset, const int32_t *__restrict__ bias,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, int8_t *__restrict__ pContextBuffer);