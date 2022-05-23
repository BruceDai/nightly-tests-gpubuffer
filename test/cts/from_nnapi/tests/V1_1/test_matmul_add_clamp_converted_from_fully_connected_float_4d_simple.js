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

  it('test matmul + add + clamp converted from fully_connected_float_4d_simple test', async function() {
    // Converted test case (from: V1_1/fully_connected_float_4d_simple.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 10]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 1, 5, 1]), [1, 2, 3, 4, 5, 6, 7, 8, -9, -10, 1, 2, 3, 4, 5, 6, 7, -8, 9, -10])};
    const op2 = builder.constant({type: 'float32', dimensions: [10, 3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 10]), [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10])});
    const b0 = builder.constant({type: 'float32', dimensions: [3]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [1, 2, 3])});
    const expected = [24, 25, 26, 58, 59, 60];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 3]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 3]), outputs.op3.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
