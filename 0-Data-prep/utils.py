import numpy as np
import os
import math
import logging
import random
import datetime
import socket
from pyproj import Geod
geod = Geod(ellps='WGS84')

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.pi = torch.acos(torch.zeros(1)).item()*2

LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO  = list(range(11))

def interpolate(t, track):
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t : 
        - track     : AIS track, whose structure is
                     [LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO]
    OUTPUT:
        - [LAT, LON, SOG, COG, HEADING, TIMESTAMP, MMSI, SHIPTYPE, LENGTH, WIDTH, CARGO]
    """
    
    before_p = np.nonzero(t >= track[:,TIMESTAMP])[0]
    after_p = np.nonzero(t < track[:,TIMESTAMP])[0]
   
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]    
        # Interpolation
        dt_full = float(track[apos,TIMESTAMP] - track[bpos,TIMESTAMP])
        if (abs(dt_full) > 2*3600):
            return None
        dt_interp = float(t - track[bpos,TIMESTAMP])
        try:
            az, _, dist = geod.inv(track[bpos,LON],
                                   track[bpos,LAT],
                                   track[apos,LON],
                                   track[apos,LAT])
            dist_interp = dist*(dt_interp/dt_full)
            lon_interp, lat_interp, _ = geod.fwd(track[bpos,LON], track[bpos,LAT],
                                               az, dist_interp)
            speed_interp = (track[apos,SOG] - track[bpos,SOG])*(dt_interp/dt_full) + track[bpos,SOG]
            course_interp = (track[apos,COG] - track[bpos,COG] )*(dt_interp/dt_full) + track[bpos,COG]
            heading_interp = (track[apos,HEADING] - track[bpos,HEADING])*(dt_interp/dt_full) + track[bpos,HEADING] 
            shiptype_interp = track[apos,SHIPTYPE] - track[bpos,SHIPTYPE]*(dt_interp/dt_full) + track[bpos,SHIPTYPE]
            length_interp = track[apos,LENGTH] - track[bpos,LENGTH]*(dt_interp/dt_full) + track[bpos,LENGTH]
            width_interp = track[apos,WIDTH] - track[bpos,WIDTH]*(dt_interp/dt_full) + track[bpos,WIDTH]
            cargo_interp = track[apos,CARGO] - track[bpos,CARGO]*(dt_interp/dt_full) + track[bpos,CARGO]
            # Not using the following interpolation methods for now
            ##############
            # rot_interp = (track[apos,ROT] - track[bpos,ROT])*(dt_interp/dt_full) + track[bpos,ROT]
            # if dt_interp > (dt_full/2):
            #     nav_interp = track[apos,NAV_STT]
            # else:
            #     nav_interp = track[bpos,NAV_STT]                             
        except:
            return None
        return np.array([lat_interp, lon_interp,
                         speed_interp, course_interp, 
                         heading_interp,t,
                         track[0,MMSI],track[0,SHIPTYPE],track[0,LENGTH],track[0,WIDTH],track[0,CARGO]])
    else:
        return None
