import datetime
import pandas as pd



class Serializer:
    def __init__(self):
        self.time = []
        self.flow = []
        self.lane_points = []
        self.observed = []

        date = datetime.datetime.now().date()
        self.filename = f"data/traffic_data_{date}.csv"

    def add_data(self, current_time_str, flow, lane_points, observed):
        self.time.append(current_time_str)
        self.flow.append(flow)
        self.lane_points.append(lane_points)
        self.observed.append(observed)
        
    def save(self):
        df = pd.DataFrame({'5 Minutes': self.time, 'Lane 1 Flow (Veh/5 Minutes)': self.flow,
                             '# Lane Points': self.lane_points, '% Observed': self.observed})
        df.to_csv(self.filename, index = False)
    
