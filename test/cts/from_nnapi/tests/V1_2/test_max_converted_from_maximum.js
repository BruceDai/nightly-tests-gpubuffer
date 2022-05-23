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

  it('test max converted from maximum_simple test', async function() {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 1, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [1.0, 0.0, -1.0, 11.0, -2.0, -1.44])};
    const input1 = builder.input('input1', {type: 'float32', dimensions: [3, 1, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [-1.0, 0.0, 1.0, 12.0, -3.0, -1.43])};
    const expected = [1.0, 0.0, 1.0, 12.0, -2.0, -1.43];
    const output0 = builder.max(input0, input1);
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test max converted from maximum_simple_relaxed test', async function() {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 1, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [1.0, 0.0, -1.0, 11.0, -2.0, -1.44])};
    const input1 = builder.input('input1', {type: 'float32', dimensions: [3, 1, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [-1.0, 0.0, 1.0, 12.0, -3.0, -1.43])};
    const expected = [1.0, 0.0, 1.0, 12.0, -2.0, -1.43];
    const output0 = builder.max(input0, input1);
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), outputs.output0.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test max converted from maximum_simple_int32 test', async function() {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'int32', dimensions: [3, 1, 2]});
    const input0Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [1, 0, -1, 11, -2, -1])};
    const input1 = builder.input('input1', {type: 'int32', dimensions: [3, 1, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [-1, 0, 1, 12, -3, -1])};
    const expected = [1, 0, 1, 12, -2, -1];
    const output0 = builder.max(input0, input1);
    const graph = builder.build({output0});
    const outputs = {output0: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]))}};
    graph.compute({'input0': input0Resource, 'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), outputs.output0.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test max converted from maximum_broadcast test', async function() {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'float32', dimensions: [3, 1, 2]});
    const input01Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [1.0, 0.0, -1.0, -2.0, -1.44, 11.0])};
    const input11 = builder.input('input11', {type: 'float32', dimensions: [2]});
    const input11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [0.5, 2.0])};
    const expected = [1.0, 2.0, 0.5, 2.0, 0.5, 11.0];
    const output01 = builder.max(input01, input11);
    const graph = builder.build({output01});
    const outputs = {output01: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]))}};
    graph.compute({'input01': input01Resource, 'input11': input11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), outputs.output01.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test max converted from maximum_broadcast_relaxed test', async function() {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'float32', dimensions: [3, 1, 2]});
    const input01Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [1.0, 0.0, -1.0, -2.0, -1.44, 11.0])};
    const input11 = builder.input('input11', {type: 'float32', dimensions: [2]});
    const input11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [0.5, 2.0])};
    const expected = [1.0, 2.0, 0.5, 2.0, 0.5, 11.0];
    const output01 = builder.max(input01, input11);
    const graph = builder.build({output01});
    const outputs = {output01: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]))}};
    graph.compute({'input01': input01Resource, 'input11': input11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), outputs.output01.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test max converted from maximum_broadcast_int32 test', async function() {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'int32', dimensions: [3, 1, 2]});
    const input01Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [1, 0, -1, -2, -1, 11])};
    const input11 = builder.input('input11', {type: 'int32', dimensions: [2]});
    const input11Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [0, 2])};
    const expected = [1, 2, 0, 2, 0, 11];
    const output01 = builder.max(input01, input11);
    const graph = builder.build({output01});
    const outputs = {output01: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]))}};
    graph.compute({'input01': input01Resource, 'input11': input11Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), outputs.output01.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
