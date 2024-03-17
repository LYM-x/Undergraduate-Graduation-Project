option = {
  xAxis: {
    type: 'category',
    data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    splitLine: {
        show: true,
    },
  },
  yAxis: {
    type: 'value',
    splitLine: {
      show: true,
    },
  },
  series: [
    {
      name:'line1',
      data: [150, 230, 224, 218, 135, 147, 260],
      type: 'line'
    },
    {
      name:'line2',
      data: [32, 230, 423, 218, 1432, 10, 26],
      type: 'line'
    },
  ],
  tooltip: {
					trigger: 'item', 
					axisPointer: {
						type: 'shadow' 
					},
					formatter: '{a} <br/>{b} : {c} <br/>' 
},
legend: {
                orient: 'horizontal',  //'vertical'
            },
};