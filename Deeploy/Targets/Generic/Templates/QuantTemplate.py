# ----------------------------------------------------------------------
#
# File: QuantTemplate.py
#
# Last edited: 12.03.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: Federico Brancasi, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _QuantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


    # def alignToContext(self, ctxt: NetworkContext,
    #                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

    #     data_out = ctxt.lookup(operatorRepresentation['data_out'])

    #     # Calculate output offset for bias hoisting
    #     output_offset = 0
    #     if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
    #         if data_out._signed == 0:
    #             print("node involved: ", operatorRepresentation)
    #         output_offset = -(data_out._signed == 0) * int(data_out.nLevels // 2)

    #     operatorRepresentation['output_offset'] = output_offset

    #     return ctxt, operatorRepresentation, []
    

    def alignToContext(self, ctxt: NetworkContext,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        scale = operatorRepresentation.get('scale', 1.0)
        
        # Scale-based heuristic
        # - scale = 128.0: Input quantization ([-1,1] → [0,255]) → NO bias hoisting
        # - scale ≈ 255.0: Activation quantization ([0,1] → [0,255]) → YES bias hoisting  
        # - scale > 200: Likely activation quantization → YES bias hoisting
        # - scale ≤ 128: Likely input quantization → NO bias hoisting
        
        if abs(scale - 128.0) < 1e-6:
            should_apply_bias_hoisting = False  # Exact 128.0 = input quantization
        elif scale > 200.0:
            should_apply_bias_hoisting = True   # High scale = activation quantization
        else:
            should_apply_bias_hoisting = False  # Low scale = input quantization
        
        # Calculate output offset for bias hoisting
        output_offset = 0
        if should_apply_bias_hoisting and hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            output_offset = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        operatorRepresentation['output_offset'] = output_offset
        return ctxt, operatorRepresentation, []

referenceTemplate = _QuantTemplate("""
// Quantization (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

    for (uint32_t i = 0; i < ${size}; i++) {

        float32_t input_val = ${data_in}[i];
        float32_t inv_scale = 1.0 / (float32_t)${scale};
        float32_t scaled_val  = (float32_t)input_val / inv_scale;      
        float32_t shifted_val = scaled_val + (float32_t)${zero_point};
        
        int32_t quantized = (int32_t)floor(shifted_val + 0.5);

        if (quantized < ${min_val}) quantized = ${min_val};
        if (quantized > ${max_val}) quantized = ${max_val};

        // Apply bias hoisting offset
        ${data_out}[i] = (${data_out_type.referencedType.typeName})(quantized + ${output_offset});

        if (i == 157410 || i == 157411 || i == 157412|| i == 157413) {
            printf("DEBUG: i=%u, input_val=%.30f, scaled_val=%.30f, quantized=%d, output_offset=%d, final_output=%d", 
                   i, input_val, scaled_val, quantized, ${output_offset}, (int)(quantized + ${output_offset}));
        }
    }

END_SINGLE_CORE
""")