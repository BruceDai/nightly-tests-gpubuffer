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

  it('test pow converted from pow test', async function() {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]), [2.0, 3.0])};
    const exponent = builder.input('exponent', {type: 'float32', dimensions: [1]});
    const exponentResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [2.0])};
    const expected = [4.0, 9.0];
    const output = builder.pow(base, exponent);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]))}};
    graph.compute({'base': baseResource, 'exponent': exponentResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1]), outputs.output.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pow converted from pow_relaxed test', async function() {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]), [2.0, 3.0])};
    const exponent = builder.input('exponent', {type: 'float32', dimensions: [1]});
    const exponentResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1]), [2.0])};
    const expected = [4.0, 9.0];
    const output = builder.pow(base, exponent);
    const graph = builder.build({output});
    const outputs = {output: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]))}};
    graph.compute({'base': baseResource, 'exponent': exponentResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 1]), outputs.output.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test pow converted from pow_2 test', async function() {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]), [2.0, 3.0])};
    const exponent1 = builder.input('exponent1', {type: 'float32', dimensions: [1, 2]});
    const exponent1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2]), [2.0, 3.0])};
    const expected = [4.0, 8.0, 9.0, 27.0];
    const output1 = builder.pow(base, exponent1);
    const graph = builder.build({output1});
    const outputs = {output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2]))}};
    graph.compute({'base': baseResource, 'exponent1': exponent1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2]), outputs.output1.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pow converted from pow_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]), [2.0, 3.0])};
    const exponent1 = builder.input('exponent1', {type: 'float32', dimensions: [1, 2]});
    const exponent1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2]), [2.0, 3.0])};
    const expected = [4.0, 8.0, 9.0, 27.0];
    const output1 = builder.pow(base, exponent1);
    const graph = builder.build({output1});
    const outputs = {output1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2]))}};
    graph.compute({'base': baseResource, 'exponent1': exponent1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2]), outputs.output1.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test pow converted from pow_3 test', async function() {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]), [2.0, 3.0])};
    const exponent2 = builder.input('exponent2', {type: 'float32', dimensions: [3, 1, 2]});
    const exponent2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [0.0, 0.5, 1.0, 2.0, 3.0, 4.0])};
    const expected = [1.0, 1.4142135623730951, 1.0, 1.7320508075688772, 2.0, 4.0, 3.0, 9.0, 8.0, 16.0, 27.0, 81.0];
    const output2 = builder.pow(base, exponent2);
    const graph = builder.build({output2});
    const outputs = {output2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 2]))}};
    graph.compute({'base': baseResource, 'exponent2': exponent2Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 2, 2]), outputs.output2.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pow converted from pow_relaxed_3 test', async function() {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 1]), [2.0, 3.0])};
    const exponent2 = builder.input('exponent2', {type: 'float32', dimensions: [3, 1, 2]});
    const exponent2Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 1, 2]), [0.0, 0.5, 1.0, 2.0, 3.0, 4.0])};
    const expected = [1.0, 1.4142135623730951, 1.0, 1.7320508075688772, 2.0, 4.0, 3.0, 9.0, 8.0, 16.0, 27.0, 81.0];
    const output2 = builder.pow(base, exponent2);
    const graph = builder.build({output2});
    const outputs = {output2: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([3, 2, 2]))}};
    graph.compute({'base': baseResource, 'exponent2': exponent2Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([3, 2, 2]), outputs.output2.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
