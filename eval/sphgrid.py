import os
import numpy as np
import scipy.integrate as spi

class SphBase():
    def __init__(self, img_h=960, img_w=1920):
        self.img_w = img_w
        self.img_h = img_h
    
    def ang2rad(self, a):
        return a/180*np.pi
    
    def rad2ang(self, r):
        return r/np.pi*180
    
    def uv2lonlat(self, u = None, v = None):
        lon = ((u+.5) - self.img_w/2.) / self.img_w * (2*np.pi) if u is not None else None
        lat = -((v+.5) - self.img_h/2.) / self.img_h * np.pi if v is not None else None
        return lon, lat
        
    def lonlat2uv(self, lon = None, lat = None):
        u = np.round(lon*self.img_w/(2*np.pi) + self.img_w/2. - .5) if lon is not None else None
        v = np.round(-lat*self.img_h/np.pi + self.img_h/2. - .5) if lat is not None else None
        return u, v
    
    def lonlat2xyz(self, lon, lat):
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(-lat)
        z = np.cos(lat) * np.cos(lon)
        return x, y, z
    
    def xyz2lonlat(self, x, y, z, norm=False):
        lon = np.arctan2(x, z)
        lat = np.arcsin(-y) if norm else np.arctan2(-y, np.sqrt(x**2 + z**2)) 
        return lon, lat   

class SphGrid(SphBase):
    def __init__(self, img_h = 960, img_w = 1920):
        super().__init__(img_h, img_w)
        self.lon_step = 1./img_w*(2*np.pi)
        self.lat_step = 1./img_h*np.pi
        
        self.v_lin = np.linspace(0, self.img_h-1, self.img_h, endpoint=False)
        self.u_lin = np.linspace(0, self.img_w-1, self.img_w, endpoint=False)

        self.lon_lin, self.lat_lin = self.uv2lonlat(u=self.u_lin, v=self.v_lin)
        self.uv_grid = np.meshgrid(self.u_lin, self.v_lin)
        
        self.init_integr_grid()
    
    def Integr_lat(self, downlim, uplim):
        def integr_helper(down_lim, up_lim):
            return spi.quad(lambda x:np.sin(x), down_lim, up_lim)
        vec_integr = np.vectorize(integr_helper)
        res, _ = vec_integr(downlim, uplim)
        return res
    
    def init_integr_grid(self):
        lat_lin_downlim = self.lat_lin - self.lat_step/2.
        lat_lin_uplim = self.lat_lin + self.lat_step/2.
        lat_lin_integr = np.abs(self.Integr_lat(lat_lin_downlim, lat_lin_uplim) * self.lon_step)
        self.integr_grid = np.repeat(lat_lin_integr[:,None], self.img_w, axis=1)
