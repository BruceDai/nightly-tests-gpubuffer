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

  it('test conv2d + add + clamp converted from conv_float_channels_weights_as_inputs test', async function() {
    // Converted test case (from: V1_0/conv_float_channels_weights_as_inputs.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), [99.0, 99.0, 99.0])};
    const op2 = builder.input('op2', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 1, 3]), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])};
    const op3 = builder.input('op3', {type: 'float32', dimensions: [3]});
    const op3Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]), [0.0, 0.0, 0.0])};
    const pad0 = 0;
    const stride = 1;
    const expected = [297.0, 594.0, 891.0];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'filterLayout': 'ohwi'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = builder.build({op4});
    const outputs = {op4: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]))}};
    graph.compute({'op1': op1Resource, 'op2': op2Resource, 'op3': op3Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 3]), outputs.op4.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
