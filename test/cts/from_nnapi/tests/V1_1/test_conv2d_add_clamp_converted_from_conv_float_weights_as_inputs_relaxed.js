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

  it('test conv2d + add + clamp converted from conv_float_weights_as_inputs_relaxed test', async function() {
    // Converted test case (from: V1_1/conv_float_weights_as_inputs_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 3, 1]), [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [0.25, 0.25, 0.25, 0.25])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [0])};
    const pad0 = 0;
    const stride = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'filterLayout': 'ohwi'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op4.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
