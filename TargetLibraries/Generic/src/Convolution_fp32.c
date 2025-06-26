/* =====================================================================
 * Title:        Convolution_float32.c
 * Description:  Float32 version of Conv2D with NCHW format (pre-padded input)
 *
 * Date:         12.05.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
 * - Calin Diaconu, University of Bologna
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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"

void Conv2d_fp32_fp32_fp32_NCHW(const float32_t *__restrict__ pSrcA, uint32_t C,
                                uint32_t H_padded, uint32_t W_padded,
                                const float32_t *__restrict__ pSrcB, uint32_t F,
                                uint32_t P, uint32_t Q, uint32_t SP,
                                uint32_t SQ, float32_t *__restrict__ pDstC,
                                const float32_t *__restrict__ bias) {
  uint32_t H_out = (H_padded - P) / SP + 1;
  uint32_t W_out = (W_padded - Q) / SQ + 1;

  for (uint32_t f = 0; f < F; ++f) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {

        float32_t sum = (bias != NULL) ? (float32_t)bias[f] : 0.0f;

        for (uint32_t c = 0; c < C; ++c) {
          for (uint32_t p = 0; p < P; ++p) {
            const uint32_t a_row =
                c * H_padded * W_padded + (h * SP + p) * W_padded;

            for (uint32_t q = 0; q < Q; ++q) {
              sum += pSrcA[a_row + (w * SQ + q)] *
                     pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
            }
          }
        }

        pDstC[f * H_out * W_out + h * W_out + w] = sum;
      }
    }
  }
}
