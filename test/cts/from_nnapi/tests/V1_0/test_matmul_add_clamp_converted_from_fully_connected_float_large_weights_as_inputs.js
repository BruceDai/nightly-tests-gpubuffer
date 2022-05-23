'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  it('test matmul + add + clamp converted from fully_connected_float_large_weights_as_inputs test', async function() {
    // Converted test case (from: V1_0/fully_connected_float_large_weights_as_inputs.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 5]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 5]), [1, 10, 100, 1000, 10000])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [5, 1]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 5]), [2, 3, 4, 5, 6])};
    const b0 = builder.input('b0', {type: 'float32', dimensions: [1]});
    const b0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [900000])};
    const expected = [965432];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'b0': b0Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1]), outputs.op3.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
