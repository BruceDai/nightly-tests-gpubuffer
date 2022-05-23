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

  it('test conv2d (fused ops) converted from depthwise_conv2d_float_large_2_relaxed test', async function() {
    // Converted test case (from: V1_1/depthwise_conv2d_float_large_2_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 100, 0.25, 1, 20, 100, 0.25, 0, 30, 100, 0.25, 1, 40, 100])});
    const op3 = builder.constant({type: 'float32', dimensions: [4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])});
    const pad0 = 0;
    const stride = 1;
    const expected = [6010, 7046, 11000, 9000];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
