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

  it('test l2Pool2d + clamp converted from l2_pool_float_relaxed test', async function() {
    // Converted test case (from: V1_1/l2_pool_float_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), [1.0, 2.0, 3.0, 4.0])};
    const pad0 = 0;
    const cons1 = 1;
    const expected = [1.0, 2.0, 3.0, 4.0];
    const interOut0 = builder.l2Pool2d(op1, {'padding': [pad0, pad0, pad0, pad0], 'strides': [cons1, cons1], 'windowDimensions': [cons1, cons1], 'layout': 'nhwc'});
    const op3 = builder.clamp(interOut0);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 1]), outputs.op3.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
