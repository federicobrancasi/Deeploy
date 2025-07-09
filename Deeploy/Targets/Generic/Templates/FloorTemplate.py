# ----------------------------------------------------------------------
#
# File: FloorTemplate.py
#
# Last edited: 03.07.2025
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

from typing import Dict

from Deeploy.DeeployTypes import NodeTemplate, OperatorRepresentation


class FloorTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: Dict, operatorRepresentation: OperatorRepresentation, **kwargs):

        data_in = ctxt['data_in']
        data_out = ctxt['data_out']

        size = operatorRepresentation['size']

        newCtxt = {
            'data_in': data_in,
            'data_out': data_out,
            'size': size,
        }

        return newCtxt


referenceTemplate = FloorTemplate("""
for(int i = 0; i < ${size}; i++) {
    // Apply floor operation, preserving the data type
    ${data_out}[i] = (${data_in_type.referencedType.typeName})floorf((float)${data_in}[i]);
}
""")
