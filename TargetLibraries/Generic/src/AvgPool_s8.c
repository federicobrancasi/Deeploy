/* =====================================================================
 * Title:        AvgPool_s8.c
 * Description:
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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"

void AvgPool2d_s8_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                          uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                          uint32_t SP, uint32_t SQ, uint32_t pool_size,
                          int8_t *__restrict__ pDstC, int32_t input_offset,
                          int32_t output_offset) {
  uint32_t H_out = (H - P) / SP + 1;
  uint32_t W_out = (W - Q) / SQ + 1;

  for (uint32_t c = 0; c < C; ++c) {
    for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
        int32_t sum = 0;

        for (uint32_t p = 0; p < P; ++p) {
          for (uint32_t q = 0; q < Q; ++q) {
            uint32_t h_in = h_out * SP + p;
            uint32_t w_in = w_out * SQ + q;
            if (h_in < H && w_in < W) {
              sum += (pSrcA[c * H * W + h_in * W + w_in] + input_offset);
            }
          }
        }

        // Divide by the fixed pool size (P*Q) and apply output offset
        // Use rounding division for better accuracy with integers
        int32_t avg = (sum + pool_size / 2) / pool_size - output_offset;

        // Clamp
        if (avg > 127)
          avg = 127;
        if (avg < -128)
          avg = -128;

        pDstC[c * H_out * W_out + h_out * W_out + w_out] = (int8_t)avg;
      }
    }
  }
}