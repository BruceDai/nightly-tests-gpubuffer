'use strict';
import * as utils from '../utils.js';

describe('test tanh', function() {
  let device;
  let context;
  before(async () => {
    const adaptor = await navigator.gpu.requestAdapter();
    device = await adaptor.requestDevice();
    context = navigator.ml.createContext(device);
  });

  async function testTanh(input, expected, shape) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: shape});
    const y = builder.tanh(x);
    const graph = builder.build({y});
    const size = utils.sizeOfShape(shape);
    const inputs = {'x': {resource: await utils.createGPUBuffer(device, size, input)}};
    const outputs = {'y': {resource: await utils.createGPUBuffer(device, size)}};
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.y, expected);
    utils.checkValue(await utils.readbackGPUBuffer(device, size, outputs.y.resource), expected);
  }
  it('tanh 1d', async function() {
    await testTanh([-1, 0, 1], [-0.76159418, 0., 0.76159418], [3]);
  });

  it('tanh 3d', async function() {
    await testTanh(
        [
          0.15102264,  -1.1556778,  -0.0657572,  -0.04362043, 1.13937,
          0.5458485,   -1.1451102,  0.3929889,   0.56226826,  -0.68606883,
          0.46685237,  -0.53841704, 0.7025275,   -1.5314125,  0.28699,
          0.84823394,  -0.18585628, -0.319641,   0.41442505,  0.88782656,
          1.0844846,   -0.56016934, 0.531165,    0.73836696,  1.0364187,
          -0.07221687, -0.9580888,  1.8173703,   -1.5682113,  -1.272829,
          2.331454,    0.2967249,   0.21472701,  -0.9332915,  2.3962052,
          0.498327,    0.53040606,  1.6241137,   0.8147571,   -0.6471784,
          0.8489049,   -0.33946696, -0.67703784, -0.07758674, 0.7667829,
          0.58996105,  0.7728692,   -0.47817922, 2.1541011,   -1.1611695,
          2.1465113,   0.64678246,  1.239878,    -0.10861816, 0.07814338,
          -1.026162,   -0.8464255,  0.53589034,  0.93667775,  1.2927296,
        ],
        [
          0.14988485,  -0.8196263,  -0.06566259, -0.04359278, 0.81420183,
          0.49740228,  -0.8161277,  0.37393406,  0.50965846,  -0.59545064,
          0.43565258,  -0.4917888,  0.60596967,  -0.910666,   0.27936205,
          0.69014573,  -0.18374546, -0.30918226, 0.39222348,  0.71031857,
          0.79485625,  -0.5081031,  0.4862711,   0.6281575,   0.7764699,
          -0.07209159, -0.74342316, 0.94857556,  -0.9167408,  -0.8545626,
          0.98129857,  0.28831258,  0.21148658,  -0.7321248,  0.9835515,
          0.4608004,   0.48569143,  0.9252187,   0.67220616,  -0.5697674,
          0.6904969,   -0.32700142, -0.5895903,  -0.07743143, 0.6450549,
          0.5298676,   0.64859474,  -0.44478422, 0.97344196,  -0.82142067,
          0.97304124,  0.56949997,  0.84542084,  -0.10819301, 0.07798471,
          -0.7723645,  -0.6891974,  0.48987082,  0.7336921,   0.85983974,
        ],
        [3, 4, 5]);
  });
});
