/* App.js */
import React, { Component } from "react";
import CanvasJSReact from '@canvasjs/react-stockcharts';
//var CanvasJSReact = require('@canvasjs/react-stockcharts');

var CanvasJS = CanvasJSReact.CanvasJS;
var CanvasJSStockChart = CanvasJSReact.CanvasJSStockChart;

class CandlestickChart extends Component {
  constructor(props) {
    super(props);
    this.state = {
      dataPoints1: [], dataPoints3: [], isLoaded: false,
      stockName: props.stockName,
      data: props.data
    };
  }

  componentDidMount() {
    this.updateChartData(this.props.data);
  }

  componentDidUpdate(prevProps) {
    if (this.props.stockName !== prevProps.stockName || this.props.data !== prevProps.data) {
      this.updateChartData(this.props.data);
    }
  }

  updateChartData(data) {
    var dps1 = [], dps3 = [];
    for (var i = 0; i < data.length; i++) {
      dps1.push({
        x: i,
        label: new Date(data[i].date).toLocaleDateString(),
        y: [
          Number(data[i].open),
          Number(data[i].high),
          Number(data[i].low),
          Number(data[i].close)
        ]
      });
      dps3.push({ x: i, y: Number(data[i].close) });
    }
    this.setState({
      isLoaded: true,
      dataPoints1: dps1,
      dataPoints3: dps3,
      stockName: this.props.stockName,
      data: this.props.data
    });
  }

  render() {
    const { dataPoints1, dataPoints3, isLoaded } = this.state;
    console.log('Candlestick selectedStock:', this.state.stockName);
    console.log('Candlestick data:', this.state.data);
    const options = {
      theme: "light2",
      title: {
        text: this.state.stockName + " Price Chart"
      },
      subtitles: [{
        text: "Price Trend"
      }],
      rangeSelector: {
        enabled: false
      },
      charts: [{
        axisX: {
          lineThickness: 5,
          tickLength: 5,
          labelFormatter: function (e) {
            return e && e.dataPoint ? e.dataPoint.label : "";
          },
          crosshair: {
            enabled: true,
            snapToDataPoint: true,
          }
        },
        axisY: {
          title: this.state.stockName,
          prefix: "$",
          tickLength: 5
        },
        toolTip: {
          shared: true
        },
        data: [{
          name: "Price (in USD)",
          yValueFormatString: "$#,###.##",
          type: "candlestick",
          risingColor: "green",
          fallingColor: "red",
          dataPoints: dataPoints1
        }]
      }],
      navigator: {
        data: [{
          dataPoints: dataPoints3
        }],
        slider: {
          minimum: Math.max(dataPoints1.length - 30, 0),
          maximum: dataPoints1.length
        }
      }
    };

    const containerProps = {
      width: "100%",
      height: "450px",
      margin: "auto"
    };

    return (
      <div>
        <div>
          {
            isLoaded &&
            <CanvasJSStockChart containerProps={containerProps} options={options} />
          }
        </div>
      </div>
    );
  }
}

export default CandlestickChart;
