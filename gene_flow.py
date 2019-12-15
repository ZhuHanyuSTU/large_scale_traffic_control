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
# ["Edge.31~21", "Edge.21~11", "Edge.11~01", "Edge.-11~01"],
routes_list = [["Edge.-11~01", "Edge.01~11", "Edge.11~21", "Edge.21~31"],
               ["Edge.00~01", "Edge.01~02"],
               ["Edge.10~11", "Edge.11~12"],
               ["Edge.20~21", "Edge.21~22"]]
flows_list = []
for idx in range(len(routes_list)):
    flow_dict = OrderedDict()
    flow_dict["vehicle"] = vehicle_paras_dict
    flow_dict["route"] = routes_list[idx]
    flow_dict["interval"] = 5.0
    flow_dict["startTime"] = 0
    flow_dict["endTime"] = -1
    flows_list.append(flow_dict)

json_str = json.dumps(flows_list, indent=2)
with open('./setting/test_flow.json', 'w') as json_file:
    json_file.write(json_str)
