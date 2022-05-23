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

  it('test instanceNormalization converted from instance_normalization_nhwc test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1])};
    const param = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [1.0, 1.0])});
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [0.0, 0.0])});
    const param2 = 0.0001;
    const layout = 'nhwc';
    const expected = [0.0, 0.0, 0.0, -0.8164898, 0.0, -0.8164898, 0.0, 1.6329796, 0.99995005, -0.6324429, -0.99995005, 1.2648858, -0.99995005, -1.2648858, 0.99995005, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = builder.build({out});
    const outputs = {out: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1])};
    const param = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [1.0, 1.0])});
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [0.0, 0.0])});
    const param2 = 0.0001;
    const layout = 'nhwc';
    const expected = [0.0, 0.0, 0.0, -0.8164898, 0.0, -0.8164898, 0.0, 1.6329796, 0.99995005, -0.6324429, -0.99995005, 1.2648858, -0.99995005, -1.2648858, 0.99995005, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = builder.build({out});
    const outputs = {out: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1])};
    const param = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [1.0, 1.0])});
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [0.0, 0.0])});
    const param2 = 0.0001;
    const layout = 'nchw';
    const expected = [0.0, 0.0, 0.0, 0.0, 0.0, -0.8164898, -0.8164898, 1.6329796, 0.99995005, -0.99995005, -0.99995005, 0.99995005, -0.6324429, 1.2648858, -1.2648858, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = builder.build({out});
    const outputs = {out: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputResource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1])};
    const param = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [1.0, 1.0])});
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [0.0, 0.0])});
    const param2 = 0.0001;
    const layout = 'nchw';
    const expected = [0.0, 0.0, 0.0, 0.0, 0.0, -0.8164898, -0.8164898, 1.6329796, 0.99995005, -0.99995005, -0.99995005, 0.99995005, -0.6324429, 1.2648858, -1.2648858, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = builder.build({out});
    const outputs = {out: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input': inputResource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nhwc_2 test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1])};
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [2.0, 2.0])});
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [10.0, 10.0])});
    const param5 = 0.0001;
    const layout = 'nhwc';
    const expected = [10.0, 10.0, 10.0, 8.367021, 10.0, 8.367021, 10.0, 13.265959, 11.9999, 8.735114, 8.0001, 12.529772, 8.0001, 7.470228, 11.9999, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = builder.build({out1});
    const outputs = {out1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out1.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nhwc_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1])};
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [2.0, 2.0])});
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [10.0, 10.0])});
    const param5 = 0.0001;
    const layout = 'nhwc';
    const expected = [10.0, 10.0, 10.0, 8.367021, 10.0, 8.367021, 10.0, 13.265959, 11.9999, 8.735114, 8.0001, 12.529772, 8.0001, 7.470228, 11.9999, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = builder.build({out1});
    const outputs = {out1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out1.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw_2 test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1])};
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [2.0, 2.0])});
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [10.0, 10.0])});
    const param5 = 0.0001;
    const layout = 'nchw';
    const expected = [10.0, 10.0, 10.0, 10.0, 10.0, 8.367021, 8.367021, 13.265959, 11.9999, 8.0001, 8.0001, 11.9999, 8.735114, 12.529772, 7.470228, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = builder.build({out1});
    const outputs = {out1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out1.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), [0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1])};
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [2.0, 2.0])});
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([]), [10.0, 10.0])});
    const param5 = 0.0001;
    const layout = 'nchw';
    const expected = [10.0, 10.0, 10.0, 10.0, 10.0, 8.367021, 8.367021, 13.265959, 11.9999, 8.0001, 8.0001, 11.9999, 8.735114, 12.529772, 7.470228, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = builder.build({out1});
    const outputs = {out1: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]))}};
    graph.compute({'input1': input1Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([2, 2, 2, 2]), outputs.out1.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
