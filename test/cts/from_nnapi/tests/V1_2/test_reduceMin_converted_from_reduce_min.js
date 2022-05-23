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

  it('test reduceMin converted from reduce_min test', async function() {
    // Converted test case (from: V1_2/reduce_min.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2]), [-1, -2, 3, 4, 5, -6])};
    const param = [-1];
    const param1 = false;
    const expected = [-2, 3, -6];
    const output0 = builder.reduceMin(input0, {'axes': param, 'keepDimensions': param1});
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]))}};
    graph.compute({'input0': input0Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test reduceMin converted from reduce_min_relaxed test', async function() {
    // Converted test case (from: V1_2/reduce_min.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2]), [-1, -2, 3, 4, 5, -6])};
    const param = [-1];
    const param1 = false;
    const expected = [-2, 3, -6];
    const output0 = builder.reduceMin(input0, {'axes': param, 'keepDimensions': param1});
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3]))}};
    graph.compute({'input0': input0Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3]), outputs.output0.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test reduceMin converted from reduce_min_2 test', async function() {
    // Converted test case (from: V1_2/reduce_min.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'float32', dimensions: [1]});
    const input01Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [9.527])};
    const param2 = [0];
    const param3 = true;
    const expected = [9.527];
    const output01 = builder.reduceMin(input01, {'axes': param2, 'keepDimensions': param3});
    const graph = builder.build({output01});
    const outputs = {output01: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]))}};
    graph.compute({'input01': input01Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1]), outputs.output01.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test reduceMin converted from reduce_min_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/reduce_min.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'float32', dimensions: [1]});
    const input01Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [9.527])};
    const param2 = [0];
    const param3 = true;
    const expected = [9.527];
    const output01 = builder.reduceMin(input01, {'axes': param2, 'keepDimensions': param3});
    const graph = builder.build({output01});
    const outputs = {output01: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]))}};
    graph.compute({'input01': input01Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1]), outputs.output01.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test reduceMin converted from reduce_min_4 test', async function() {
    // Converted test case (from: V1_2/reduce_min.mod.py)
    const builder = new MLGraphBuilder(context);
    const input03 = builder.input('input03', {type: 'float32', dimensions: [4, 3, 2]});
    const input03Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 3, 2]), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])};
    const param6 = [0, 2];
    const param7 = true;
    const expected = [0.1, 0.3, 0.5];
    const output03 = builder.reduceMin(input03, {'axes': param6, 'keepDimensions': param7});
    const graph = builder.build({output03});
    const outputs = {output03: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1]))}};
    graph.compute({'input03': input03Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 1]), outputs.output03.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test reduceMin converted from reduce_min_relaxed_4 test', async function() {
    // Converted test case (from: V1_2/reduce_min.mod.py)
    const builder = new MLGraphBuilder(context);
    const input03 = builder.input('input03', {type: 'float32', dimensions: [4, 3, 2]});
    const input03Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4, 3, 2]), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])};
    const param6 = [0, 2];
    const param7 = true;
    const expected = [0.1, 0.3, 0.5];
    const output03 = builder.reduceMin(input03, {'axes': param6, 'keepDimensions': param7});
    const graph = builder.build({output03});
    const outputs = {output03: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 3, 1]))}};
    graph.compute({'input03': input03Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 3, 1]), outputs.output03.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
