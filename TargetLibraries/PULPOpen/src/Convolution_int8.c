/* =====================================================================
 * Title:        Convolution_int8.c
 * Description:  Int version of Conv2D with NCHW format (pre-padded input)
 *
 * Date:         23.01.2025
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

 #include "DeeployPULPMath.h"
 #include "pmsis.h"
 
 void Conv2d_s8_s8_s32_HWC(
     const int8_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
     const int8_t *__restrict__ pSrcB, uint32_t F, uint32_t P, uint32_t Q,
     uint32_t SP, uint32_t SQ, int32_t *__restrict__ pDstC, int32_t input_offset,
     int32_t output_offset, const int32_t *__restrict__ bias, uint32_t pad_top,
     uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
 
   const uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
   const uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
   
   /* 
   * FBRANCASI: This dummy variable and its modification 
   * inside the loop prevent aggressive optimizations 
   * by the compiler that can cause a heisenbug. 
   */
   uint32_t dummy = 0;
 
   for (uint32_t f_idx = 0; f_idx < F; f_idx++) {
     const int32_t bias_val = (bias != NULL) ? bias[f_idx] : 0;
     
     for (uint32_t h_idx = 0; h_idx < H_out; h_idx++) {
       for (uint32_t w_idx = 0; w_idx < W_out; w_idx++) {
         int32_t result = bias_val;
         
         for (uint32_t p_idx = 0; p_idx < P; p_idx++) {
           int32_t h_in = h_idx * SP + p_idx - pad_top;
           
           if (h_in < 0 || h_in >= (int32_t)H) {
             continue;
           }
           
           for (uint32_t q_idx = 0; q_idx < Q; q_idx++) {
             int32_t w_in = w_idx * SQ + q_idx - pad_left;
             
             if (w_in < 0 || w_in >= (int32_t)W) {
               continue;
             }
             
             dummy = dummy + 1;
             
             for (uint32_t c_idx = 0; c_idx < C; c_idx++) {
               uint32_t input_idx = (h_in * W + w_in) * C + c_idx;
               uint32_t weight_idx = (f_idx * P * Q * C) + (p_idx * Q * C) + (q_idx * C) + c_idx;
               
               int8_t input_val = pSrcA[input_idx];
               int8_t weight_val = pSrcB[weight_idx];
               
               result += ((int32_t)input_val + input_offset) * (int32_t)weight_val;
             }
           }
         }
         
         result += output_offset;
         
         uint32_t output_idx = (h_idx * W_out + w_idx) * F + f_idx;
         pDstC[output_idx] = result;
       }
     }
   }
 }