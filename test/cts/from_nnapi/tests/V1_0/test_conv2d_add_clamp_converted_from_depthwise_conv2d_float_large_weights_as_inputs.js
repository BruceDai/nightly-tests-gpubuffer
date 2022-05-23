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

  it('test conv2d + add + clamp converted from depthwise_conv2d_float_large_weights_as_inputs test', async function() {
    // Converted test case (from: V1_0/depthwise_conv2d_float_large_weights_as_inputs.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 21, 10, 22, 10, 23, 10, 24])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [2]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])};
    const pad0 = 0;
    const stride = 1;
    const expected = [110, 246];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'groups': 2, 'filterLayout': 'ihwo'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
