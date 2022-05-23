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

  it('test matmul + add + clamp converted from fully_connected_v1_2 test', async function() {
    // Converted test case (from: V1_2/fully_connected_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1]), [2, 32, 16])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1]), [2])});
    const b0 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [4])});
    const expected = [8, 68, 36];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1]), outputs.op3.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test matmul + add + clamp converted from fully_connected_v1_2_relaxed test', async function() {
    // Converted test case (from: V1_2/fully_connected_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [3, 1]});
    const op1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1]), [2, 32, 16])};
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1]), [2])});
    const b0 = builder.constant({type: 'float32', dimensions: [1]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [4])});
    const expected = [8, 68, 36];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1]))}};
    graph.compute({'op1': op1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1]), outputs.op3.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
