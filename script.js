let dataset = [];
let labels = [];
let basicInputs, basicLabels, basicInputTensor, basiclabelTensor, basicNMInputs;
let inputs, inputTensor, labelTensor, nmInputs;
let inputs2, values2, inputTensor2, labelTensor2, labelMin2, labelMax2, nmInputs2, nmLabels2;
let model;
let price_scale;

const width = 1000;
const height = 600;
const nodeSize = 10;
const color = d3.scaleOrdinal(d3.schemeCategory10);
let networkGraph;
let x,y;
let x_h, y_h;
let groups;
let margin;
let width_d, height_d;

let synth_easy, synth_hard;
let isInitialSetup = true;
let samples;

export {drawGraph, drawGraph2, model, samples, x_h, y_h, margin, nmInputs, model2};

import { CustomCallback, trainModel, createModel, nodesValue, hiddenLayersValue} from './model2.js';
import {buildNodeGraph3, draw3, linearModel, linearModel2, networkGraph3, trainLinearModel, trainLinearModel2} from './linearregression.js';
import {generatePredictionMatrix, predictZones, updateHeatmap} from './heatmap.js';
import {normalRandom, classifyTwoGaussData, classifyCircleData} from './dataset.js';
import {CustomCallback2, trainModel2, createComplexModel, draw2, buildNodeGraph2, hiddenLayersValue2, nodesValue2, networkGraph2, model2} from './complexmodel.js';



async function loadData() {
    let url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv';
    const response = await d3.csv(url).then((data) => {
        dataset = data;
        console.log(data);
        // set the dimensions and margins of the graph
        margin = {top: 10, right: 30, bottom: 30, left: 60},
        width_d = 460 - margin.left - margin.right,
        height_d = 400 - margin.top - margin.bottom;

        // append the svg object to the body of the page
        var svg = d3.selectAll("#heatmap")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        price_scale = d3.scaleLinear([0, 1], ["red", "blue"])


        // Data for linear regression
        synth_easy = classifyTwoGaussData(640, .1);
        basicInputs = synth_easy.map(d =>[d.x, d.y]);
        basicLabels = synth_easy.map(d => d.label);
        basicInputTensor = tf.tensor2d(basicInputs, [basicInputs.length, 2]);
        basiclabelTensor = tf.tensor2d(basicLabels, [basicLabels.length, 1]);
        basicNMInputs = normalize_data(basicInputTensor, 0);
        // Data for complex synth interactable
        synth_hard = classifyCircleData(640, .1)
        labels = synth_hard.map(d => d.label);
        inputs = synth_hard.map(d => [d.x, d.y]);
        let col1 = inputs.map(d => d[0]);
        let col2 = inputs.map(d => d[1]);
        inputTensor = tf.tensor2d(inputs, [inputs.length, 2]);
        labelTensor = tf.tensor2d(labels, [labels.length, 1]);
        nmInputs = normalize_data(inputTensor, 0);
        let labels2 = data.map(d => parseFloat(d.medv));
        inputs2 = data.map(d => [parseFloat(d.crim), parseFloat(d.rm), parseFloat(d.zn), parseFloat(d.indus), parseFloat(d.chas), parseFloat(d.nox), 
            parseFloat(d.age), parseFloat(d.dis), parseFloat(d.rad), parseFloat(d.tax), parseFloat(d.ptratio), parseFloat(d.b), parseFloat(d.lstat)]);
        console.log(inputs2)
        values2 = data.map(extractData);
        inputTensor2 = tf.tensor2d(inputs2, [inputs2.length, 13]);
        labelTensor2 = tf.tensor2d(labels2, [labels2.length, 1]);
        labelMin2 = labelTensor2.min();
        labelMax2 = labelTensor2.max();
        nmInputs2 = normalize_data(inputTensor2, 0);
        console.log(nmInputs2)
        nmLabels2 = labelTensor2.sub(labelMin2).div(labelMax2.sub(labelMin2));
        x = d3.scaleLinear()
        .domain([0, 1])
        .range([ 0, width_d ]);
        svg.append("g")
        .attr("transform", "translate(0," + height_d + ")")
        .call(d3.axisBottom(x));

        // Add Y axis
        y = d3.scaleLinear()
        .domain([0, 1])
        .range([ height_d, 0]);
        svg.append("g")
        .call(d3.axisLeft(y));

        // Add dots
        svg.append('g')
        .selectAll("dot")
        .data(synth_hard)
        .enter()
        .append("circle")
        .attr("cx", function (d) { return x((parseFloat(d.x)-Math.min(...col1))/(Math.max(...col1)-Math.min(...col1))); } )
        .attr("cy", function (d) { return y((parseFloat(d.y)-Math.min(...col2))/(Math.max(...col2)-Math.min(...col2))); } )
        .attr("r", 1.5)
        .style("fill", function (d) { return price_scale(d.label)})
        
        let svg2 = d3.select("#actual_heatmap")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");
        groups = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95]
        x_h = d3.scaleBand()
        .range([ 0, width_d ])
        .domain(groups)
        .padding(0.01);
        // Add X axis to the second heatmap
        svg2.append("g")
        .attr("transform", "translate(0," + height_d + ")")
        .call(d3.axisBottom(x_h));

        // Build Y scales and axis for the second heatmap:
        y_h = d3.scaleBand()
        .range([ height_d, 0 ])
        .domain(groups)
        .padding(0.01);
        // Add Y axis to the second heatmap
        svg2.append("g")
        .call(d3.axisLeft(y_h));

        d3.select("#epochSlider").on("change", function(d){
            selectedValue = this.value
            console.log(selectedValue)
          })
        generate_scatter1();
        generate_heatmap();
        generate_heatmap2();
        // Draw the initial neural network graph
        draw();
        draw2();
        draw3();
    });
}

window.addEventListener('DOMContentLoaded', (event) => {
    loadData();
    samples = generatePredictionMatrix();
    document.getElementById('tbutton').addEventListener('click', function () {
        console.log(nmInputs.arraySync())
        draw()
        trainModel(model, nmInputs, labelTensor);
    });

    document.getElementById('tbutton2').addEventListener('click', function () {
        trainModel2(model2, nmInputs2, nmLabels2);
    });

    document.getElementById('lbutton1').addEventListener('click', function () {
        trainLinearModel(linearModel, basicNMInputs, basiclabelTensor);
    });
    document.getElementById('lbutton2').addEventListener('click', function () {
        trainLinearModel2(linearModel2, nmInputs, labelTensor);
    });
    // Event listener for changes in the number of hidden layers
    // document.getElementById("hiddenLayersDropdown").addEventListener("change", function () {
    //     hiddenLayersValue = parseInt(this.value);
    //     draw();
    // });

    // Event listener for changes in the number of nodes
    document.getElementById("nodesDropdown1").addEventListener("change", function () {
        nodesValue[0] = parseInt(this.value);
        draw();
    })
    document.getElementById("nodesDropdown2").addEventListener("change", function () {
        nodesValue[1] = parseInt(this.value);
        draw();
    });

    const epochSlider = document.getElementById('epochSlider');
    epochSlider.value = 1;

    const epochValueSpan = document.getElementById('epochValue');
    epochValueSpan.textContent = "Epoch 1";

    // Update the content of the span element with the initial value of the slider
    epochValueSpan.textContent = epochSlider.value;

    // Add input event listener to the epochSlider
    epochSlider.addEventListener('input', function() {
        // Update the content of the span element with the current value of the slider
        epochValueSpan.textContent = "Epoch: " + this.value;
    });
});

function normalize_data(input, feat) {
    let inputMin = input.min(feat);
    let inputMax = input.max(feat);
    return input.sub(inputMin).div(inputMax.sub(inputMin));
}

function unnormalize_data(normed, input, feat) {
    inputMin = input.min(feat);
    console.log(inputMin.reshape([2, 1]).print());
    inputMax = input.max(feat);
    return normed.mul(inputMax.sub(inputMin).reshape([2, 1])).add(inputMin.reshape([2, 1]));
}


function extractData(d) {
    return { x: parseFloat(d.crim), y: parseFloat(d.medv) };
}

function mapXY(value, index) {
    return { x: value, y: unY[index] };
}



function buildNodeGraph() {
    let newGraph = {
        "nodes": []
    };

    // Construct input layer
    let newFirstLayer = [];
    for (let i = 0; i < 2; i++) {
        let newTempLayer = { "label": "i" + i, "layer": 1 };
        newFirstLayer.push(newTempLayer);
    }

    // Construct hidden layers
    let hiddenLayers = [];
    for (let hiddenLayerLoop = 0; hiddenLayerLoop < hiddenLayersValue; hiddenLayerLoop++) {
        let newHiddenLayer = [];
        // For the height of this hidden layer
        for (let i = 0; i < nodesValue[hiddenLayerLoop]; i++) {
            let newTempLayer = { "label": "h" + hiddenLayerLoop + i, "layer": (hiddenLayerLoop + 2) };
            newHiddenLayer.push(newTempLayer);
        }
        hiddenLayers.push(newHiddenLayer);
    }

    // Construct output layer
    let newOutputLayer = [];
    for (let i = 0; i < 1; i++) {
        let newTempLayer = { "label": "o" + i, "layer": hiddenLayersValue + 2 };
        newOutputLayer.push(newTempLayer);
    }

    // Add to newGraph
    let allMiddle = newGraph.nodes.concat.apply([], hiddenLayers);
    newGraph.nodes = newGraph.nodes.concat(newFirstLayer, allMiddle, newOutputLayer);

    console.log(newGraph);
    return newGraph;
}


function drawGraph(networkGraph, svg) {
    let graph = networkGraph;
    let nodes = graph.nodes;

    // Get network size
    let netsize = {};
    nodes.forEach(function (d) {
        if (d.layer in netsize) {
            netsize[d.layer] += 1;
        } else {
            netsize[d.layer] = 1;
        }
        d["lidx"] = netsize[d.layer];
    });

    // Calculate distances between nodes
    let largestLayerSize = Math.max.apply(
        null, Object.keys(netsize).map(function (i) { return netsize[i]; }));

    let xdist = width / (Object.keys(netsize).length + 1),
        ydist = (height - 15) / largestLayerSize;

    // Create node locations
    nodes.map(function (d) {
        d["x"] = (d.layer) * xdist;
        d["y"] = (((d.lidx - 0.5) + ((largestLayerSize - netsize[d.layer]) / 2)) * ydist) + 10;
    });

    // Autogenerate links
    let links = [];
    let pos = 0;
    nodes.map(function (d, i) {
        for (let n in nodes) {
            if (d.layer == nodes[n].layer - 1) {
                links.push({ "source": parseInt(i), "target": parseInt(n), "value": 1, "overall":pos});
                pos = pos+1;
            }
        }
    }).filter(function (d) { return typeof d !== "undefined"; });

    // Draw links
    let link = svg.selectAll(".link")
        .data(links, d => d.source + "-" + d.target) // Use a unique identifier for each link
        .enter().append("line")
        .on('mouseover', (event, d) => {
            const tooltip = d3.select('#tooltip');
            tooltip.transition()
            .duration(200)
            .style('position', 'absolute')
            .style('background-color', 'white')
            .style('padding', '6px')
            .style('border', '1px solid #ccc')
            .style('border-radius', '5px')
            .style('font-size', '12px');
    
            // Set tooltip content
            tooltip.html(`${d.value}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        })
        .on('mousemove', function (event) {
            d3.select('#tooltip')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        })
        .on('mouseout', () => {
            d3.select('#tooltip')
            .transition()
            .duration(250)
            .style('opacity', 0);

            console.log("moved")
        })
        .attr("class", "link")
        .attr("x1", d => nodes[d.source].x)
        .attr("y1", d => nodes[d.source].y)
        .attr("x2", d => nodes[d.target].x)
        .attr("y2", d => nodes[d.target].y)
        .style("stroke", "#999")
        .style("stroke-opacity", 1)
        .style("stroke-width", d => 4*d.value);

    // Draw nodes
    let node = svg.selectAll(".node")
        .data(nodes)
        .enter().append("g")
        .attr("transform", function (d) {
            return "translate(" + d.x + "," + d.y + ")";
        });

    let circle = node.append("circle")
        .attr("class", "node")
        .attr("r", nodeSize)
        .style("fill", function (d) { return color(d.layer); });

    node.append("text")
        .attr("dx", "-.35em")
        .attr("dy", ".35em")
        .attr("font-size", ".6em");
}

function drawGraph2(networkGraph, svg) {
    let graph = networkGraph;
    let nodes = graph.nodes;

    // Calculate the spacing between nodes in each layer
    let xdist = width / 2; // Half the width for two layers
    let ydist = height / 3; // Divide the height into three equal parts

    // Create node locations for each layer
    nodes.forEach(function (d, i) {
        if (i < 2) {
            d["x"] = xdist; // Position nodes in the middle for the first layer
            d["y"] = (i + 1) * ydist; // Position nodes vertically for the first layer
            d["layer"] = 1; // Assign layer 1 to the first two nodes
        } else {
            // Position the third node equidistant from both circles in the other layer
            d["x"] = xdist + 200; 
            d["y"] = (nodes[0].y + nodes[1].y) / 2; 
            d["layer"] = 2; // Assign layer 2 to the third node
        }
    });

    // Autogenerate links between nodes
    let links = [];
    // Connect the first two nodes in the first layer to the third node in the second layer
    links.push({ "source": 0, "target": 2, "value": 1 });
    links.push({ "source": 1, "target": 2, "value": 1 });

    // Draw links
    let link = svg.selectAll(".link")
        .data(links)
        .enter().append("line")
        .attr("class", "link")
        .attr("x1", d => nodes[d.source].x)
        .attr("y1", d => nodes[d.source].y)
        .attr("x2", d => nodes[d.target].x)
        .attr("y2", d => nodes[d.target].y)
        .style("stroke", "#999")
        .style("stroke-opacity", .7)
        .style("stroke-width", d => 4 * d.value);

    // Draw nodes
    let node = svg.selectAll(".node")
        .data(nodes)
        .enter().append("g")
        .attr("transform", function (d) {
            return "translate(" + d.x + "," + d.y + ")";
        });

    let circle = node.append("circle")
        .attr("class", "node")
        .attr("r", nodeSize)
        .style("fill", function (d) { return color(d.layer); });

    node.append("text")
        .attr("dx", "-.35em")
        .attr("dy", ".35em")
        .attr("font-size", ".6em");
}


function draw() {
    if (isInitialSetup) {
        let svg = d3.select("#neuralNet").append("svg")
            .attr("width", width)
            .attr("height", height);
        networkGraph = buildNodeGraph();
        drawGraph(networkGraph, svg);
        isInitialSetup = false;
    } else {
        let svg = d3.select("#neuralNet").select("svg")
        svg.selectAll("*").remove()
        console.log("drawing   " + new Date());
        networkGraph = buildNodeGraph();
        drawGraph(networkGraph, svg);
    }
    model = createModel();
}

function generate_scatter1(){
    let col1 = basicInputs.map(d => d[0]);
    let col2 = basicInputs.map(d => d[1]);

    var svg = d3.select("#basic_scatter")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    x = d3.scaleLinear()
        .domain([0, 1])
        .range([ 0, width_d ]);
        svg.append("g")
        .attr("transform", "translate(0," + height_d + ")")
        .call(d3.axisBottom(x));

    y = d3.scaleLinear()
        .domain([0, 1])
        .range([ height_d, 0]);
        svg.append("g")
        .call(d3.axisLeft(y));

        // Add dots
    svg.append('g')
        .selectAll("dot")
        .data(synth_easy)
        .enter()
        .append("circle")
        .attr("cx", function (d) { return x((parseFloat(d.x)-Math.min(...col1))/(Math.max(...col1)-Math.min(...col1))); } )
        .attr("cy", function (d) { return y((parseFloat(d.y)-Math.min(...col2))/(Math.max(...col2)-Math.min(...col2))); } )
        .attr("r", 1.5)
        .style("fill", function (d) { return price_scale(d.label)})
    
}

function generate_heatmap(){
    let svg2 = d3.select("#basic_heatmap")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");
    groups = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95]
    x_h = d3.scaleBand()
    .range([ 0, width_d ])
    .domain(groups)
    .padding(0.01);
    // Add X axis to the second heatmap
    svg2.append("g")
    .attr("transform", "translate(0," + height_d + ")")
    .call(d3.axisBottom(x_h));

    // Build Y scales and axis for the second heatmap:
    y_h = d3.scaleBand()
    .range([ height_d, 0 ])
    .domain(groups)
    .padding(0.01);
    // Add Y axis to the second heatmap
    svg2.append("g")
    .call(d3.axisLeft(y_h));
}

function generate_heatmap2(){
    let svg2 = d3.select("#complex_heatmap")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");
    groups = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95]
    x_h = d3.scaleBand()
    .range([ 0, width_d ])
    .domain(groups)
    .padding(0.01);
    // Add X axis to the second heatmap
    svg2.append("g")
    .attr("transform", "translate(0," + height_d + ")")
    .call(d3.axisBottom(x_h));

    // Build Y scales and axis for the second heatmap:
    y_h = d3.scaleBand()
    .range([ height_d, 0 ])
    .domain(groups)
    .padding(0.01);
    // Add Y axis to the second heatmap
    svg2.append("g")
    .call(d3.axisLeft(y_h));
}
