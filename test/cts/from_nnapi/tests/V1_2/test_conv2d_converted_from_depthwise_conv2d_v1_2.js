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

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 21, 10, 22, 10, 23, 10, 24])};
    const op22 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])});
    const op32 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])});
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nhwc';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 21, 10, 22, 10, 23, 10, 24])};
    const op22 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])});
    const op32 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])});
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nhwc';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 21, 10, 22, 10, 23, 10, 24])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [2]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])};
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nhwc';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 21, 10, 22, 10, 23, 10, 24])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [2]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])};
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nhwc';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 2]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24])};
    const op22 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])});
    const op32 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])});
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nchw';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24])};
    const op22 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])});
    const op32 = builder.constant({type: 'float32', dimensions: [2]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])});
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nchw';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]))}};
    graph.compute({'op12': op12Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [2]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])};
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nchw';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op12Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24])};
    const op22 = builder.input('op22', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op22Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 2]), [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1])};
    const op32 = builder.input('op32', {type: 'float32', dimensions: [2]});
    const op32Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([2]), [100, 200])};
    const param13 = 0;
    const param14 = 0;
    const param15 = 0;
    const param16 = 0;
    const param17 = 1;
    const param18 = 1;
    const layout = 'nchw';
    const expected = [110, 246];
    const op42 = builder.conv2d(op12, op22, {'bias': op32, 'padding': [param15, param16, param13, param14], 'strides': [param18, param17], 'inputLayout': layout, 'groups': 2, 'filterLayout': 'ihwo'});
    const graph = builder.build({op42});
    const outputs = {op42: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]))}};
    graph.compute({'op12': op12Resource, 'op22': op22Resource, 'op32': op32Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 2, 1, 1]), outputs.op42.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0])};
    const op23 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])});
    const op33 = builder.constant({type: 'float32', dimensions: [4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])});
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nhwc';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0])};
    const op23 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])});
    const op33 = builder.constant({type: 'float32', dimensions: [4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])});
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nhwc';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_weight_as_input_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [4]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])};
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nhwc';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nhwc_weight_as_input_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [4]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])};
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nhwc';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 1, 1, 4]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 4, 2, 2]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0])};
    const op23 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])});
    const op33 = builder.constant({type: 'float32', dimensions: [4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])});
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nchw';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 4, 2, 2]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0])};
    const op23 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])});
    const op33 = builder.constant({type: 'float32', dimensions: [4]}, {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])});
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nchw';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]))}};
    graph.compute({'op13': op13Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_weight_as_input_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 4, 2, 2]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [4]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])};
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nchw';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]), outputs.op43.resource), expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_v1_2_large_nchw_weight_as_input_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/depthwise_conv2d_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 4, 2, 2]});
    const op13Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 2, 2]), [10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0])};
    const op23 = builder.input('op23', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op23Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 2, 2, 4]), [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50])};
    const op33 = builder.input('op33', {type: 'float32', dimensions: [4]});
    const op33Resource = {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([4]), [6000, 7000, 8000, 9000])};
    const param21 = 0;
    const param22 = 0;
    const param23 = 0;
    const param24 = 0;
    const param25 = 1;
    const param26 = 1;
    const layout = 'nchw';
    const expected = [6010, 7046, 11000, 9000];
    const op43 = builder.conv2d(op13, op23, {'bias': op33, 'padding': [param23, param24, param21, param22], 'strides': [param26, param25], 'inputLayout': layout, 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = builder.build({op43});
    const outputs = {op43: {resource: await utils.createGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]))}};
    graph.compute({'op13': op13Resource, 'op23': op23Resource, 'op33': op33Resource}, outputs);
    utils.checkValue(await utils.readbackGPUBuffer(device, utils.sizeOfShape([1, 4, 1, 1]), outputs.op43.resource), expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
