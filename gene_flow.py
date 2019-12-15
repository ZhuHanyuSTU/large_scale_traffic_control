from collections import defaultdict, OrderedDict
import json

vehicle_paras_dict = {
      "length": 5.0,
      "width": 2.0,
      "maxPosAcc": 2.0,
      "maxNegAcc": 4.5,
      "usualPosAcc": 2.0,
      "usualNegAcc": 4.5,
      "minGap": 2.5,
      "maxSpeed": 16.67,
      "headwayTime": 1.5
    }

flow_dict = OrderedDict()
flow_dict["vehicle"] = vehicle_paras_dict


{
    "vehicle": {
      "length": 5.0,
      "width": 2.0,
      "maxPosAcc": 2.0,
      "maxNegAcc": 4.5,
      "usualPosAcc": 2.0,
      "usualNegAcc": 4.5,
      "minGap": 2.5,
      "maxSpeed": 16.67,
      "headwayTime": 1.5
    },
    "route": [
      "-gneE11",
      "gneE10"
    ],
    "interval": 5.0,
    "startTime": 0,
    "endTime": -1
  }

json_str = json.dumps(flow_dict, indent=4)
with open('test_data.json', 'w') as json_file:
    json_file.write(json_str)