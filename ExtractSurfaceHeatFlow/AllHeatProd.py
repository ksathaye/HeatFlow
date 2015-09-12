import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import interp1d as interp1
from scipy.interpolate import interp2d as interp2
from scipy.interpolate import griddata as griddata
import os
from scipy.interpolate import NearestNDInterpolator as Nearest2D
from mpl_toolkits.basemap import Basemap
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
import timeit
import cPickle as pickle
import cv2
from scipy import spatial
from sklearn.cluster import DBSCAN
from matplotlib.path import Path
from scipy.spatial import ConvexHull

#%%
def LinearSolveAll():
    Dir=os.getcwd();
    DataDir=Dir + '/DataFormatted/';
    Locations=np.genfromtxt(DataDir+'SeismicLocations.csv');
    Locations[:,0]=Locations[:,0]-360;
    Density=np.genfromtxt(DataDir+'DenseAll.csv');
    Qs=np.genfromtxt(DataDir+'LongLatSurfaceHeat.csv',skip_header=1,delimiter=',');
    Qm=np.genfromtxt(DataDir+'MantleHeat.txt',skip_header=1,delimiter=',');
    QsInterp=Nearest2D(Qs[:,0:2],Qs[:,2]);
    QmInterp=Nearest2D(Qm[:,0:2],Qm[:,2]);
    
    Avocado=6.022e23; # mols to atoms conversion

    qs=QsInterp(Locations[:,0:2])/1000;
    qm=QmInterp(Locations[:,0:2])/1000;
    #Density[Density>3.1]=np.nan;
    
    Fels=(3-Density)/(0.3);
    Fels[Density<2.7]=1;
    Fels[Density>3]=0;
    years=365.24*24*60*60;#years to seconds conversion

    Depth=np.genfromtxt(DataDir+'Depth.csv');
    dz=(Depth[1]-Depth[0])*1000;
    
    UContentU=2.8e-6/238; #upper crust uranium mass fraction
    ThContentU=UContentU*3.8/232; #upper crust thorium mass fraction
    K40ContentU=2*120e-6*3.4e-2/94; #upper crust thorium mass fraction
    
    UContentL=0.2e-6/238; #mol/g of each cell
    ThContentL=1.2e-6/232;
    K40ContentL=2*120e-6*0.6e-2/94;
    
    alpha238=7.41e-12;#Joules/decay
    alpha235=7.24e-12;#Joules/decay
    alpha232=6.24e-12;#Joules/decay
    beta=1.14e-13; #Joules/decay
    
    LamU238 = np.log(2)/(4.468*1e9);#% decay rate of U in years
    LamTh232 = np.log(2)/(1.405e10); # decay rate of Th in years
    LamU235 = np.log(2)/(703800000); #decay rate of 235U in years
    LamK40=np.log(2)/1.248e9;#decay rate of K40 in years
    
    UraniumHeatL=alpha238*Avocado*UContentL*LamU238/years+alpha235*Avocado*UContentL*LamU235/years/137.88;
    ThoriumHeatL=alpha232*Avocado*ThContentL*LamTh232/years;
    KHeatL=beta*Avocado*K40ContentL*LamK40/years;
    TotalHeatL=UraniumHeatL+ThoriumHeatL+KHeatL; # W/gram
    
    UraniumHeatU=alpha238*Avocado*UContentU*LamU238/years+alpha235*Avocado*UContentU*LamU235/years/137.88;
    ThoriumHeatU=alpha232*Avocado*ThContentU*LamTh232/years;
    KHeatU=beta*Avocado*K40ContentU*LamK40/years;
    
    qc=qs-qm;
    FluxL=np.nansum((1-Fels)*TotalHeatL*dz*Density*1e6,0);
    TotalHeatU=(qc-FluxL)/np.nansum(Fels*Density*1e6*dz,0);
    
    print(Density)
    plt.close('all')
    #plt.plot(Density,Depth)
    #plt.show();
    
#%%
def LatLongMap():
    UpperHeatDiff=LinearSolveAll();

    plt.close('all');
    latcorners = [34,40];
    loncorners = [-108,-100];
    lon_0 = -105;
    lat_0 = 35;
    Dir=os.getcwd();
    DataDir=Dir + '/DataFormatted/';
    Density=np.genfromtxt(DataDir+'DenseAll.csv');
    Depth=np.genfromtxt(DataDir+'Depth.csv');
    dz=(Depth[1]-Depth[0])*1000;

    Locations=np.genfromtxt(DataDir+'SeismicLocations.csv');
    Locations[:,0]=Locations[:,0]-360;
    Qs=np.genfromtxt(DataDir+'LongLatSurfaceHeat.csv',skip_header=1,delimiter=',');
    Qm=np.genfromtxt(DataDir+'MantleHeat.txt',skip_header=1,delimiter=',');
    QsInterp=Nearest2D(Qs[:,0:2],Qs[:,2]);
    QmInterp=Nearest2D(Qm[:,0:2],Qm[:,2]);
    Avocado=6.022e23; # mols to atoms conversion

    qs=QsInterp(Locations[:,0:2])/1000;
    qm=QmInterp(Locations[:,0:2])/1000;
    
    Temp70All=np.loadtxt('Temp_70km_estimates.txt');
    Temp70Near=Nearest2D(Temp70All[:,0:2],Temp70All[:,2]);
    Temp70SeismicLoc=Temp70Near(Locations[:,0],Locations[:,1]);

    Density[Depth>70]=np.nan;
    plt.close('all')
    plt.plot(Density,Depth,c='k',lw=0.5)
    plt.gca().invert_yaxis();
    #plt.scatter(Locations[:,0],Locations[:,1],c=Temp70SeismicLoc,s=50,edgecolor='none')
    LLH=np.zeros([len(Locations),3]);
    LLH[:,0:2]=Locations[:,0:2];
    LLH[:,2]=Temp70SeismicLoc;
    #BaseMapPoints(LLH,'Temp70Seismic.png')
    
    #plt.close('all'); 
    kOliv=np.loadtxt('ForsterThermCond.csv',delimiter=',',skiprows=1);
    #kOliv=kOliv[,:];
    #plt.plot(kOliv[:,0],kOliv[:,1],marker='o')
    P2=np.polyfit(kOliv[:,0],kOliv[:,1],2);
    k2=np.linspace(np.min(kOliv[:,0]),np.max(kOliv[:,0]));
    P2P=np.polyval(P2,k2);
    #plt.plot(k2,P2P,c='k');
    
    #plt.show();
    plt.close('all')
    if 1==0:

        plt.subplot(1,3,1)
        m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                rsphere=6371200.,resolution='l',area_thresh=10000)
        m.drawcoastlines();
        LocationsCoord=m(Locations[:,0],Locations[:,1]);
        m.drawstates(linewidth=2);
        m.drawcountries(linewidth=2);
        m.scatter(LocationsCoord[0],LocationsCoord[1],c=qs*1e3,s=50,edgecolor='none')
        cb = plt.colorbar(orientation='horizontal',ticks=[40,50,60,70,80,90,100])
        meridians = np.arange(180.,360.,2.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        parallels = np.arange(0.,90,2.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        plt.title('Surface Heat Flux mW/m$^2$',fontsize=10)
        cb.set_clim(39, 101);
        NMtext=m(-107.5,35);
        plt.text(NMtext[0],NMtext[1],'NM')
        COtext=m(-107,39);
        plt.text(COtext[0],COtext[1],'CO')
        KStext=m(-101.5,39);
        plt.text(KStext[0],KStext[1],'KS')
        TXtext=m(-101.5,35)
        plt.text(TXtext[0],TXtext[1],'TX')
    
        plt.subplot(1,3,2)
        m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                rsphere=6371200.,resolution='l',area_thresh=10000)
        m.drawcoastlines();
        LocationsCoord=m(Locations[:,0],Locations[:,1]);
        m.drawstates(linewidth=2);
        m.drawcountries(linewidth=2);
        m.scatter(LocationsCoord[0],LocationsCoord[1],c=qm*1e3,s=50,edgecolor='none')
        cb = plt.colorbar(orientation='horizontal',ticks=[27,30,33,36,39])
        cb.set_clim(int(np.nanmin(qm)),int(39.0));
        
        
        meridians = np.arange(180.,360.,2.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        parallels = np.arange(0.,90,2.)
        m.drawparallels(parallels,labels=[0,0,0,0],fontsize=10)
        plt.title('Mantle Heat Flux mW/m$^2$',fontsize=10)
        NMtext=m(-107.5,35);
        plt.text(NMtext[0],NMtext[1],'NM')
        COtext=m(-107,39);
        plt.text(COtext[0],COtext[1],'CO')
        KStext=m(-101.5,39);
        plt.text(KStext[0],KStext[1],'KS')
        TXtext=m(-101.5,35)
        plt.text(TXtext[0],TXtext[1],'TX')
    
        plt.subplot(1,3,3)
        m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                rsphere=6371200.,resolution='l',area_thresh=10000)
        m.drawcoastlines();
        LocationsCoord=m(Locations[:,0],Locations[:,1]);
        m.drawstates(linewidth=2);
        m.drawcountries(linewidth=2);
        #m.scatter(LocationsCoord[0],LocationsCoord[1],c=UpperHeatDiff,s=50,edgecolor='none')
        print(type(UpperHeatDiff))
        print(np.shape(UpperHeatDiff))
        print('UpperHeatDiff')
        cb = plt.colorbar(orientation='horizontal',ticks=[0.5,1,1.5,2,2.5]);
        meridians = np.arange(180.,360.,2.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        parallels = np.arange(0.,90,2.)
        m.drawparallels(parallels,labels=[0,0,0,0],fontsize=10)
        cb.set_clim(0.5,2.6);
        cb.ax.set_yticklabels(['0.5','1.0','1.5','2.0','2.5']);    
        plt.title('Fraction of Expected Production',fontsize=10)
        NMtext=m(-107.5,35);
        plt.text(NMtext[0],NMtext[1],'NM')
        KStext=m(-101.5,39);
        plt.text(KStext[0],KStext[1],'KS')
        TXtext=m(-101.5,35)
        plt.text(TXtext[0],TXtext[1],'TX')
        COtext=m(-107,39);
        plt.text(COtext[0],COtext[1],'CO')

        plt.savefig('LatLongMapHeatFlux.pdf',format='pdf', dpi=1000)
    #plt.show();
   
def LatLongXY(Origin,Points):
    
    Lat=Points[:,0]*np.pi/180;
    Long=Points[:,1]*np.pi/180;
    a=6378137;
    b=6356752.3142;
    e2=(a**2-b**2)/(a**2);
    DeltaLat=111132.954-559.822*np.cos(2*Lat)+1.175*np.cos(4*Lat);
    DeltaLong=a*np.pi*np.cos(Lat)/(180*np.sqrt(1-e2*(np.sin(Lat)**2)));
    
    LatDiff=Points[:,0]-40;
    LongDiff=Points[:,1]+100;
    
    EastNorth=np.zeros([len(LatDiff),2]);
    EastNorth[:,1]=DeltaLat*LatDiff;
    EastNorth[:,0]=DeltaLong*LongDiff;
    
    return EastNorth;
    
def XYLatLong(Origin,Points):
    
    phi=np.linspace(np.deg2rad(20),np.deg2rad(60),100);
    yphi=(111132-559*np.cos(2*phi)+1.175*np.cos(4*phi))*(np.rad2deg(phi)-Origin[1]);

    yInterp=interp1(yphi,phi);
    Lat=yInterp(Points[:,1]);
    a=6378137;
    b=6356752.3142;
    e2=(a**2-b**2)/(a**2);

    P=np.pi*a*np.cos(Lat)/(180*np.sqrt(1-e2*(np.sin(Lat)**2)));
    Lat=np.rad2deg(Lat);
    Long=Points[:,0]/P+Origin[0];
    
    LongLat=np.zeros([len(Lat),2]);
    LongLat[:,0]=Long;
    LongLat[:,1]=Lat;
    
    return LongLat;

def PlotInterp():    
    EastNorthFlux=PlotBores();
    FluxEst=FluxKriging(EastNorthFlux);
    FluxEstLongLat=XYLatLong([-100,40],FluxEst);
    BoreData=np.genfromtxt('SMUData.csv',delimiter=',',skip_header=1);
    BoreData[BoreData==-1000]=np.nan;
    BoreData[BoreData[:,2]<0,2]=np.nan;
    BoreData[BoreData[:,2]>150,2]=np.nan;
    BoreData[BoreData[:,0]<25,:]=np.nan;
    BoreData[BoreData[:,0]>50,:]=np.nan;

    if 0==1:
        m = Basemap(projection='merc',llcrnrlat=25,urcrnrlat=50,\
                llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='c')
        m.drawcoastlines();
        m.drawcountries();
        m.drawstates();
    
        m.drawmapboundary(fill_color='w')
        #m.fillcontinents(color='w',lake_color='aqua');
        parallels = np.arange(0.,81,10.)
        m.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(10.,351.,20.)
        m.drawmeridians(meridians,labels=[True,False,False,True])
        LocationsCoord=m(BoreData[:,1],BoreData[:,0]);
        LocationsInterp=m(FluxEstLongLat[:,0],FluxEstLongLat[:,1]);
        m.scatter(LocationsInterp[0],LocationsInterp[1],s=10,edgecolor='none',c=FluxEst[:,2]);
        #m.scatter(LocationsCoord[0],LocationsCoord[1],s=50,edgecolor='none',c=BoreData[:,2]);
        cb = plt.colorbar(orientation='horizontal')
        cb.set_clim(0,100);

        #plt.show();
        tic=timeit.default_timer();
        plt.savefig('PlotInterp.png',dpi=100,format='png');
        toc=timeit.default_timer();

        print(-tic+toc)
    return FluxEst
    
def GetBaselines(LongOrLat):
    plt.close('all')
    if LongOrLat==01:
        img = cv2.imread('HeatMapLat.jpg');
    elif LongOrLat==0:
        img = cv2.imread('HeatMapMeridians.jpg');
    #img[sum(img,2)==255*3]=np.nan;
    [b,g,r] = cv2.split(img);
    #b[sum(img,2)==255*3]=np.nan;
    g=g.astype(float);
    r=r.astype(float);
    b=b.astype(float);
    ImSum=b+r+g;
    ImShape=np.shape(img);
    x=np.arange(ImShape[1]);
    y=np.arange(ImShape[0]);
    XX,YY=np.meshgrid(x,y);
    
    Meridians=ImSum.astype(float);
    Meridians[ImSum>0]=np.nan;    
    MerArray=np.zeros([len(np.ravel(Meridians)),3]);

    MerArray[:,0]=np.ravel(XX);
    MerArray[:,1]=np.ravel(YY);
    MerArray[:,2]=np.ravel(Meridians);
    
    CountMer=np.count_nonzero(~np.isnan(MerArray[:,2]));
    MerArray2=np.zeros([CountMer,2]);
    MerArray2[:,0]=MerArray[np.isnan(MerArray[:,2])==0,0];
    MerArray2[:,1]=MerArray[np.isnan(MerArray[:,2])==0,1];
    #MerArray2[:,2]=MerArray[np.isnan(MerArray[:,2])==0,2];
    
    TestLength= len(MerArray2);
    print(len(MerArray2))
    tic=timeit.default_timer();
    tree=spatial.cKDTree(MerArray2[0:TestLength,0:2]); 
    toc=timeit.default_timer();
    print(1e3*(toc-tic))
    
    tic=timeit.default_timer();
    Near=tree.query_pairs(1.5,1);
    toc=timeit.default_timer();
    print(1e3*(toc-tic))
    tic=timeit.default_timer();
    L=list(Near);
    L=np.array(L);
    L=L[L[:,0].argsort(),]
    L=np.array(L);
    toc=timeit.default_timer();
    print(1e3*(toc-tic))
    Uni=np.unique(L);
    NumNear=np.bincount(np.ravel(L),minlength=len(Uni));    

    MeridianPoints=MerArray2[NumNear>2,:];
        
    return MeridianPoints;

def QuadLat49():
    
    img = cv2.imread('HeatMapLat.jpg');
    
    [b,g,r] = cv2.split(img);
    b=b.astype(float);
    g=g.astype(float);
    r=r.astype(float);
    ImSum=b+r+g;
    b[ImSum==255*3]=np.nan;
    r[ImSum==255*3]=np.nan;
    g[ImSum==255*3]=np.nan;

    ImShape=np.shape(img);
    x=np.arange(ImShape[1]);
    y=np.arange(ImShape[0]);
    XX,YY=np.meshgrid(x,y);

    ImSum[ImSum>0]=np.nan;
    ImSum[ImSum==0]=1;

    Lat49=np.zeros([np.nansum(ImSum),2]);
    Lat49[:,0]=XX[ImSum>0]
    Lat49[:,1]=YY[ImSum>0]
    
    #plt.close();
    #plt.imshow(ImSum);
    #plt.scatter(Lat49[:,0],Lat49[:,1])
    P=np.polyfit(Lat49[:,0],Lat49[:,1],2);
    Quad49=np.polyval(P,x);
    #plt.plot(x,Quad49)
    #plt.show();

    return P,Quad49;
    
def PixelScale():
    
    img = cv2.imread('Km1000Scale.jpg');
    [b,g,r] = cv2.split(img);
    b=b.astype(float);
    g=g.astype(float);
    r=r.astype(float);
    ImSum=b+r+g;
    Blk=np.where(ImSum==0);
    xAx=Blk[1];
    PixTokm=1/(1e-3*(np.max(xAx)-np.min(xAx)));
    PixTokm=PixTokm*1.05;
    return PixTokm
    
def PixToLatLong(RParallel,xMeet,x49):
    MerLongs=np.genfromtxt('MeridianLongs.csv',skip_header=1,usecols=1,delimiter=',');
    PixTokm=PixelScale();
    
    MerLong1=MerLongs[0];
    MerLong2=MerLongs[-1];
    
    PixtoLong=(MerLong2-MerLong1)/(x49[1]-x49[0]);
    Pixb=MerLong2-PixtoLong*x49[1];
    
    Rkm=RParallel*PixTokm;
    phi=np.deg2rad(np.linspace(0,90,1000));
    yphi=-((111132-559*np.cos(2*phi)+1.175*np.cos(4*phi)))*1e-3*(np.rad2deg(phi)-49);
    yInterp=interp1(yphi,np.rad2deg(phi));
    Lat=yInterp(Rkm);
    
    Long=PixtoLong*xMeet+Pixb;
    LongLat=np.zeros([len(Long),2]);

    LongLat[:,0]=Long;
    LongLat[:,1]=Lat;
    
    return LongLat;

def BaseMapPoints(LLH,fn):
    
    #numPoints=200;
    #LongVec=np.linspace(np.nanmin(LLH[:,0]),np.nanmax(LLH[:,0]),numPoints);
    #LatVec=np.linspace(25,50,numPoints);
    
    #GridX,GridY=np.meshgrid(LongVec,LatVec);
    
    #A=np.isnan(LLH);
    #B=np.sum(A,1);
    #LLH=LLH[B==0];

    #HeatInterp=Nearest2D(LLH[:,0:2],LLH[:,2]);
    #Inter=HeatInterp(GridX,GridY);
    
    #InH=ConvexHull(LLH[:,0:2]);
    
    plt.close('all');
    m = Basemap(projection='merc',llcrnrlat=25,urcrnrlat=50,llcrnrlon=-125,urcrnrlon=-65,lat_ts=20,resolution='h')
    m.drawcoastlines();
    m.drawcountries();
    m.drawstates();
    LocationsCoord=m(LLH[:,0],LLH[:,1]);
    meridians = np.arange(-125,-65,10.);
    parallels = np.arange(25,50,5);
    
    m.drawparallels(parallels,labels=[1,1,0,0]);
    m.drawmeridians(meridians,labels=[0,0,0,1]);

    m.drawstates(linewidth=1);
    m.drawcountries(linewidth=2);
    m.scatter(LocationsCoord[0],LocationsCoord[1],s=5,edgecolor='none',c=LLH[:,2]);
    plt.colorbar(orientation='horizontal');
    if fn[0]=='H':
        plt.clim(20,100);
        plt.title('Surface Heat Flux (mW/m$^2$)')
    else:
        plt.title('Temperature at 70km ($^{\circ}$C)')
    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    plt.savefig(fn,format='png')

def NewtonLatitude(Poly,x0,y0):
    if 1==0:
        img = cv2.imread('HeatMapLat.jpg');
        
        [b,g,r] = cv2.split(img);
        b=b.astype(float);
        g=g.astype(float);
        r=r.astype(float);
        ImSum=b+r+g;
        b[ImSum==255*3]=np.nan;
        r[ImSum==255*3]=np.nan;
        g[ImSum==255*3]=np.nan;
    
        ImShape=np.shape(img);
        xx=np.arange(ImShape[1]);
        yy=np.arange(ImShape[0]);
        XX,YY=np.meshgrid(xx,yy);
    
        x0=np.ravel(XX);
        y0=np.ravel(YY);

    x=x0;

    A=Poly[0];
    B=Poly[1];
    C=Poly[2];
    it=4;
    for i in range(it):
        
        R=np.sqrt((x-x0)**2+(np.polyval(Poly,x)-y0)**2);
        dRdx=((x-x0)+(np.polyval(Poly,x)-y0)*(2*A*x+B))/R;
        HI=(x-x0)+(np.polyval(Poly,x)-y0)*(2*A*x+B);
        dHI=1+2*A*(np.polyval(Poly,x)-y0)+(2*A*x+B)**2;
        d2Rdx2=(R*dHI-HI*dRdx)/(R**2);

        x=x-dRdx/d2Rdx2;
    R=np.sqrt((x-x0)**2+(np.polyval(Poly,x)-y0)**2);
    
    return x,R

def LatLongQuad():
    LongBase=GetBaselines(0);
    PixToKM=PixelScale();
    #MeridianBase=PixToLatLong();
    img = cv2.imread('HeatMapSMU.jpg');
    Poly,Quad49=QuadLat49();
    [b,g,r] = cv2.split(img);
    b=b.astype(float);
    g=g.astype(float);
    r=r.astype(float);
    ImSum=b+r+g;
    b[ImSum==255*3]=np.nan;
    r[ImSum==255*3]=np.nan;
    g[ImSum==255*3]=np.nan;

    ImShape=np.shape(img);
    x=np.arange(ImShape[1]);
    y=np.arange(ImShape[0]);
    XX,YY=np.meshgrid(x,y);
    
    x0=np.ravel(XX);
    y0=np.ravel(YY);
    x0=x0.astype(float);
    y0=y0.astype(float);
    b0=np.ravel(b);
    
    x0[np.isnan(b0)]=np.nan;
    x01000=x0[0:-1:10];
    y01000=y0[0:-1:10];
    x49Meet,RParallel=NewtonLatitude(Poly,x0,y0);
    
    db = DBSCAN(min_samples=100,eps=50).fit(LongBase);
    labels = db.labels_
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels);
    
    BottomPoints=np.zeros([n_clust+2,2]);
    TopPoints=np.zeros([n_clust+2,2]);
    for i in range(n_clust):
        MeridPoints=LongBase[labels==i,:];
        BottomInd=np.argmin(MeridPoints[:,1]);
        BottomPoints[i+1,0]=MeridPoints[BottomInd,0];
        BottomPoints[i+1,1]=MeridPoints[BottomInd,1];

        TopInd=np.argmax(MeridPoints[:,1]);
        TopPoints[i+1,0]=MeridPoints[TopInd,0];
        TopPoints[i+1,1]=MeridPoints[TopInd,1];
    
    m=np.zeros(2);
    k=np.zeros(2);
    
    m[1]=(TopPoints[2,1]-BottomPoints[2,1])/(TopPoints[2,0]-BottomPoints[2,0]);
    m[0]=(TopPoints[1,1]-BottomPoints[1,1])/(TopPoints[1,0]-BottomPoints[1,0]);
    
    k[1]=TopPoints[2,1]-m[1]*TopPoints[2,0];
    k[0]=TopPoints[1,1]-m[0]*TopPoints[1,0];
    
    x49=np.zeros(2);
    Intersect49xOpt=np.zeros(2);
    
    Intersect49xOpt[0]=(-(Poly[1]-m[0])+np.sqrt((Poly[1]-m[0])**2-4*Poly[0]*(Poly[2]-k[0])))/(2*Poly[0]);
    Intersect49xOpt[1]=(-(Poly[1]-m[0])-np.sqrt((Poly[1]-m[0])**2-4*Poly[0]*(Poly[2]-k[0])))/(2*Poly[0]);
    x49[0]=np.min(np.abs(Intersect49xOpt));

    Intersect49xOpt[0]=(-(Poly[1]-m[1])+np.sqrt((Poly[1]-m[1])**2-4*Poly[0]*(Poly[2]-k[1])))/(2*Poly[0]);
    Intersect49xOpt[1]=(-(Poly[1]-m[1])-np.sqrt((Poly[1]-m[1])**2-4*Poly[0]*(Poly[2]-k[1])))/(2*Poly[0]);
    x49[1]=np.min(np.abs(Intersect49xOpt));
    
    xDiff=x49Meet-x0;
    
    LongLat=PixToLatLong(RParallel,x49Meet,x49);
    Long=LongLat[:,0];
    #Lat=LongLat[:,1];
    #plt.close('all')
    #plt.scatter(x01000,y01000,s=50,edgecolor='none',c=Long[0:-1:10]);
    #plt.plot(x,Quad49);
    #plt.plot([TopPoints[2,0],BottomPoints[2,0]],[TopPoints[2,1],BottomPoints[2,1]] ,c='k')
    #plt.plot([TopPoints[1,0],BottomPoints[1,0]],[TopPoints[1,1],BottomPoints[1,1]],c='k')
    #plt.colorbar();
    #plt.imshow(img)
    #plt.scatter(x49Meet,x0)
    #plt.plot([0,3500],[0,3500],c='red',lw=3)
    #plt.show();
    
    return LongLat;


    for i in range(len(PolyX)):

        #plt.plot(PolyX[i,:],PolyY[i,:],lw=2,c='k');
        PX=np.transpose(PolyX[i,:])
        PY=np.transpose(PolyY[i,:])
        PXY=[PX, PY];
        PXY=np.transpose(PXY);
        p=Path(PXY);
        ZoneBool=p.contains_points(np.transpose([np.ravel(XX),np.ravel(YY)]));
        Zones[ZoneBool==1,2]=i+1;
        InPoly=Zones[Zones[:,2]==i+1,:];
        if i<2:
            x1=TopPoints[1,0];
            y1=TopPoints[1,1];
            x2=BottomPoints[1,0];
            y2=BottomPoints[1,1];
            m=(y2-y1)/(x2-x1);
            b=y2-m*x2;
            
        else: 
            x1=TopPoints[i,0];
            y1=TopPoints[i,1];
            x2=BottomPoints[i,0];
            y2=BottomPoints[i,1];
            m=(y2-y1)/(x2-x1);
            b=y2-m*x2;
        
        RMerid=np.zeros(len(Zones[ZoneBool==1]));
        x0=Zones[ZoneBool==1,0];
        y0=Zones[ZoneBool==1,1];
        RMerid=np.abs(((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/np.sqrt((y2-y1)**2+(x2-x1)**2));
        Intersect=np.zeros([len(Zones[ZoneBool==1]),2]);
        Intersect[:,0]=(x0+m*y0-m*b)/(m*m+1);
        Intersect[:,1]=m*((x0+m*y0-m*b)/(m*m+1))+b;
        
        Intersect49xOpt=np.zeros(2);
        Intersect49xOpt[0]=(-(PolyQuad[1]-m)+np.sqrt((PolyQuad[1]-m)**2-4*PolyQuad[0]*(PolyQuad[2]-b)))/(2*PolyQuad[0]);
        Intersect49xOpt[1]=(-(PolyQuad[1]-m)-np.sqrt((PolyQuad[1]-m)**2-4*PolyQuad[0]*(PolyQuad[2]-b)))/(2*PolyQuad[0]);
    
        Intersect49x=np.min(np.abs(Intersect49xOpt));
        Intersect49y=np.polyval(PolyQuad,Intersect49x);
        #RParallel=np.sqrt((Intersect[:,1]-Intersect49y)**2+(Intersect[:,0]-Intersect49x)**2);
        RParallel=NewtonLatitude(PolyQuad,x0,y0);
        RMeridkm=RMerid*PixToKM;
        RParallelkm=RParallel*PixToKM;
        points=np.zeros([len(RMeridkm),2]);
        points[:,0]=RMeridkm;
        points[:,1]=RParallelkm;
        #LatLongPoints=PixToLatLong(points,i);
        #LatLong[ZoneBool==1,0]=LatLongPoints[:,0];
        #LatLong[ZoneBool==1,1]=LatLongPoints[:,1];
        #LatLong[ZoneBool==1,2]=i;

        #if i<2:
        #LatLong1000=LatLong[0:-1:1000,:];
        x01000=x0[0:-1:10];
        y01000=y0[0:-1:10];
        plt.scatter(x01000,y01000,s=100,edgecolor='none',c=RParallel[0:-1:10])
        #BaseMapPoints(LatLong1000);

        
    plt.imshow(img);
    plt.plot(x,Quad49);
    #LongBase2D=npimg.zeros([len(x),len(y)]);
    #LongBase2D[LongBase[:,0].astype(int),LongBase[:,1].astype(int)]=1;
    #[MeridianConnect,B]=ndimage.label(LongBase);
    #print(B)
    plt.colorbar();
    #plt.show();

    return RParallel;

def GetColors():
    imgAll = cv2.imread('HeatMapSMU.jpg');
    [bAll,gAll,rAll] = cv2.split(imgAll);
    bAll=bAll.astype(float);
    gAll=gAll.astype(float);
    rAll=rAll.astype(float);
    
    bAll=np.ravel(bAll);
    rAll=np.ravel(rAll);
    gAll=np.ravel(gAll);

    ImSum=(bAll)+(gAll)+(rAll);

    bAll[ImSum==255*3]=np.nan;
    rAll[ImSum==255*3]=np.nan;
    gAll[ImSum==255*3]=np.nan;

    img = cv2.imread('ColormapSMU.jpg');
    [b,g,r] = cv2.split(img);
    L=np.arange(len(r));
    b=b.astype(float);
    g=g.astype(float);
    r=r.astype(float);

    b=b[:,20];
    g=g[:,20];
    r=r[:,20];
    
    plt.close('all')
    #plt.imshow(img)
    b[np.abs(np.diff(b))>.1]=np.nan;
    r[np.abs(np.diff(r))>.1]=np.nan;
    g[np.abs(np.diff(g))>.1]=np.nan;
    Colors=(b+r+g)*0;
    b=Colors+b;
    r=Colors+r;
    g=Colors+g;
    
    ColTup=np.zeros([len(b),3]);
    ColTup[:,2]=r;
    ColTup[:,1]=g;
    ColTup[:,0]=b;
    #ColTup=tuple(ColTup);
    
    UniqCol,NumUniq=np.unique(ColTup,return_counts=True);
    
    T = np.ascontiguousarray(ColTup).view(np.dtype((np.void, ColTup.dtype.itemsize * ColTup.shape[1])))
    idx = np.unique(T, return_index=True,return_counts=True);
    counts=idx[2];
    counts[counts>300]=1;
    idx=idx[1];
    idx=idx[counts>20];
    idx=np.sort(idx);
    uZ = ColTup[idx,:];

    ColorHeat=np.zeros(len(idx));
    ColorHeat[0:18]=np.linspace(15,100,18);
    ColorHeat[18]=110;
    ColorHeat[19]=120;
    ColorHeat[20]=150;
    ColorHeat=ColorHeat[::-1];
    
    idx=np.sort(idx);
    
    ColorsAll=np.zeros([len(bAll),3]);
    ColorsAll[:,0]=bAll;
    ColorsAll[:,1]=gAll;
    ColorsAll[:,2]=rAll;
    
    ColorsDist=np.zeros([len(bAll),21]);
    
    for i in range(len(uZ)):
        ColorsDist[:,i]=np.sqrt((bAll-uZ[i,0])**2+(gAll-uZ[i,1])**2+(rAll-uZ[i,2])**2);
    
    #ColorsDist[np.isnan(ColorsDist)]=1000;
    #ColorsDist[ColorsDist>20]=np.nan;
    
    #CD=ColorsDist[ColorsDist[:,0]>-1,:];
    CD=ColorsDist;
    A=np.zeros([len(CD),2]);
    A[:,0]=np.min(CD,1);
    A[:,1]=ColorHeat[np.argmin(CD,1)];
    A[A[:,0]>20,:]=np.nan;
    if 1==0:
        plt.close('all')
        plt.subplot(1,2,1)
        plt.scatter(ColorHeat,idx,c=uZ/255 ,s=150,edgecolor='none')
        plt.gca().invert_yaxis()
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        plt.subplot(2,1,1)
        plt.hist(A[np.isnan(A[:,0])==0,1],bins=21); 
        #plt.xlim([0,20]); 
    
        plt.subplot(2,1,2)
        uZ2=np.zeros(np.shape(uZ));
        uZ2[:,0]=uZ[:,2];
        uZ2[:,2]=uZ[:,0];
        uZ2[:,1]=uZ[:,1];
        plt.scatter(np.arange(len(ColorHeat)),ColorHeat,c=uZ2/255 ,s=200,edgecolor='none')
        plt.xlim([0,20]); 
        plt.show();

    return A[:,1];
    
def Temp70km():
    Temp=np.loadtxt('Temp_70km_estimates.txt');
    BaseMapPoints(Temp,'TempMap70.png');
    print(len(Temp))
    return Temp;

def SaveLatLongHeat():
    LongLat=LatLongQuad();
    SurfHeatFlux=GetColors();
    
    LongLatHeat=np.zeros([len(SurfHeatFlux),3]);
    LongLatHeat[:,0:2]=LongLat;
    LongLatHeat[:,2]=SurfHeatFlux;
    A=np.isnan(LongLatHeat);
    B=np.sum(A,1);
    LongLatHeatNoNaN=LongLatHeat[B==0,:];

    #LongLatHeatN=LongLatHeat[np.isnan(LongLatHeat[:,2])==0,:];
    #np.savetxt('LongLatSurfaceHeat.csv',LongLatHeatN);
    Jump=1000;
    y=LongLatHeat[0:-1:Jump,1];
    x=LongLatHeat[0:-1:Jump,0];
    H=LongLatHeat[0:-1:Jump,2];
    
    Temp=np.loadtxt('Temp_70km_estimates.txt');
    HeatInterp=Nearest2D(LongLatHeatNoNaN[:,0:2],LongLatHeatNoNaN[:,2]);
    
    LLHReduce=np.zeros([len(Temp),3]);
    LLHReduce[:,1]=Temp[:,1];
    LLHReduce[:,0]=Temp[:,0];
    LLHReduce[:,2]=HeatInterp(Temp[:,0],Temp[:,1]);
    np.savetxt('LongLatSurfaceHeat.csv',LLHReduce,delimiter=',');
    if 1==0:
        plt.scatter(x,y,s=20,edgecolor='none',c=H);
        plt.xlim([-125,-65])
        plt.ylim([25,50])
        plt.colorbar();
        #plt.clim(20,100);
        plt.savefig('HeatMapProc.png',format='png')
        plt.show();

    return LLHReduce;
    
LLH=SaveLatLongHeat();
BaseMapPoints(LLH,'HeatMapUSAr.png');
#Temp=Temp70km();
#LatLongMap();