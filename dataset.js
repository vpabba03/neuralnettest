function normalRandom(mean = 0, variance = 1) {
    let v1, v2, s
    do {
      v1 = 2 * Math.random() - 1;
      v2 = 2 * Math.random() - 1;
      s = v1 * v1 + v2 * v2;
    } while (s > 1);

    let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
    return mean + Math.sqrt(variance) * result;
  }

function randUniform(a, b) {
    return Math.random() * (b - a) + a;
}

function dist(a, b) {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

function classifyTwoGaussData(numSamples, noise){
    let points = [];

    let varianceScale = d3.scaleLinear().domain([0, .5]).range([0.5, 4]);
    let variance = varianceScale(noise);

    function genGauss(cx, cy, label) {
        for (let i = 0; i < numSamples / 2; i++) {
            let x = normalRandom(cx, variance);
            let y = normalRandom(cy, variance);
            points.push({x, y, label});
        }
    }

    genGauss(2, 2, 1); // Gaussian with positive examples.
    genGauss(-2, -2, 0); // Gaussian with negative examples.
    return points;
}

function classifyCircleData(numSamples, noise) {
  let points = [];
  let radius = 5;
  function getCircleLabel(p, center) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : 0;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}

export {normalRandom, classifyTwoGaussData, classifyCircleData};