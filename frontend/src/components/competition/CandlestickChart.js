import React from 'react';
import ReactApexChart from 'react-apexcharts';
import dayjs from 'dayjs';

class CandlestickChart extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      series: [{
        name: 'candle',
        data: props.data
      }],
      options: {
        chart: {
          height: 350,
          type: 'candlestick',
        },
        title: {
          text: props.stockName,
          align: 'top'
        },
        tooltip: {
          enabled: true,
        },
        xaxis: {
          type: 'category',
          labels: {
            formatter: function (val) {
              return dayjs(val).format('YYYY-MM-DD');
            }
          },
          tickAmount: 10
        },
        yaxis: {
          tooltip: {
            enabled: true
          }
        }
      }
    };
  }

  componentDidUpdate(prevProps) {
    if (this.props.data !== prevProps.data || this.props.stockName !== prevProps.stockName) {
      this.setState({
        ...this.state,
        series: [{
          name: 'candle',
          data: this.props.data
        }],
        options: {
          ...this.state.options,
          title: {
            ...this.state.options.title,
            text: this.props.stockName
          }
        }
      });
    }
  }

  render() {
    return (
      <div>
        <div id="chart">
          <ReactApexChart options={this.state.options} series={this.state.series} type="candlestick" height={350} />
        </div>
      </div>
    );
  }
}

export default CandlestickChart;
