import {x_h, y_h, margin} from './script.js';

function generatePredictionMatrix() {
    let prediction_matrix = []
    for (let z = .05; z<1; z+=.1) {
      for (let x = .05; x<1; x+=.1) {
        prediction_matrix.push([z, x])
      }
    }
    let predictionMatrix = tf.tensor2d(prediction_matrix, [prediction_matrix.length, 2]);
    return predictionMatrix
  }

async function predictZones(predictionMatrix, model) {
    const predictions = model.predict(predictionMatrix);
    const values = predictions.dataSync();
    const coords = predictionMatrix.dataSync()
    const visualizationData = [];
    // Loop through each prediction
    let j=0
    for (let i = 0; i < values.length; i ++) {
        let predictionValue;
        if (values[i]>.5){
            predictionValue = 1
        }
        else{
            predictionValue = 0;
        }
        visualizationData.push({ value: predictionValue, X: +(coords[j].toFixed(2)), Y: +(coords[j+1].toFixed(2))});
        j+=2
    }

    return visualizationData;
}

// Define the updateHeatmap function
async function updateHeatmap(predictionResults) {
    // Select the SVG element where the heatmap will be rendered
    const svg = d3.select("#actual_heatmap").select('svg');
    // Remove any existing rects
    svg.selectAll("rect").remove();
    const to_scale = predictionResults.map(d => d.value);
    console.log(predictionResults)
    const heat_scale = d3.scaleLinear([0, 1], ["red", "blue"])
    // Create rects for each element in the prediction matrix grid
    const rects = svg.selectAll("rect")
        .data(predictionResults)
        .enter().append("rect")
        .attr("x", d => x_h((d.X)))
        .attr("y", d => y_h((d.Y)))
        .attr("width", x_h.bandwidth)
        .attr("height", y_h.bandwidth)
        .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")")
        .style("fill", d=>heat_scale(d.value));
    console.log(rects)
}

async function updateBasicHeatmap(predictionResults) {
    // Select the SVG element where the heatmap will be rendered
    const svg = d3.select("#basic_heatmap").select('svg');
    // Remove any existing rects
    svg.selectAll("rect").remove();
    const to_scale = predictionResults.map(d => d.value);
    console.log(predictionResults)
    const heat_scale = d3.scaleLinear([0, 1], ["red", "blue"])
    // Create rects for each element in the prediction matrix grid
    const rects = svg.selectAll("rect")
        .data(predictionResults)
        .enter().append("rect")
        .attr("x", d => x_h((d.X)))
        .attr("y", d => y_h((d.Y)))
        .attr("width", x_h.bandwidth)
        .attr("height", y_h.bandwidth)
        .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")")
        .style("fill", d=>heat_scale(d.value));
    console.log(rects)
}

async function updateBasicHeatmap2(predictionResults) {
    // Select the SVG element where the heatmap will be rendered
    const svg = d3.select("#complex_heatmap").select('svg');
    // Remove any existing rects
    svg.selectAll("rect").remove();
    const to_scale = predictionResults.map(d => d.value);
    console.log(predictionResults)
    const heat_scale = d3.scaleLinear([0, 1], ["red", "blue"])
    // Create rects for each element in the prediction matrix grid
    const rects = svg.selectAll("rect")
        .data(predictionResults)
        .enter().append("rect")
        .attr("x", d => x_h((d.X)))
        .attr("y", d => y_h((d.Y)))
        .attr("width", x_h.bandwidth)
        .attr("height", y_h.bandwidth)
        .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")")
        .style("fill", d=>heat_scale(d.value));
    console.log(rects)
}

export {generatePredictionMatrix, predictZones, updateHeatmap, updateBasicHeatmap, updateBasicHeatmap2};