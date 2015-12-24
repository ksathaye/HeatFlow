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
import pykrige.kriging_tools as kt
import timeit
import cPickle as pickle
import cv2
from scipy import spatial
from matplotlib.path import Path
from scipy.spatial import ConvexHull

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
    
    print(TotalHeatL)
    print(dz)
    plt.close('all')
    return qc*1e3 #return in W/g
    

def Temp70km(Locations):
    Dir=os.getcwd();
    DataDir=Dir + '/DataFormatted/';
    Temp70=np.loadtxt(DataDir+'Temp_70km_estimates.csv',delimiter=',');
    Temp70Interp=Nearest2D(Temp70[:,0:2],Temp70[:,2]);
    Temp70Out=Temp70Interp(Locations[:,0],Locations[:,1]);

    return Temp70Out;

def KFels(T):
    A=0.64;
    B=1200;
    k=1+A+B/(350+T);
    return k;
#%%thermal conductivity of mafic lower crust
def KMafic(T):
    A=1.7;
    B=474;
    k=1+A+B/(350+T);
    return k;
#%%averaging function for middle crust
def KMix(X,T):
    KF=KFels(T);
    KM=KMafic(T);
    k=KF*X+(1-X)*KM;
    k[X==0]=KMant(T[X==0]);
    return k;

def KMant(T):
    TP=np.array([543, 632, 812, 1006, 1204])-273;
    kP=[4.14216 ,3.40996 ,3.20076 ,2.8242, 2.21752];
    P=np.polyfit(TP,kP,1);
    k=np.polyval(P,T);
    k=k*0+3;
    return k;

def HeatProd():

    Dir=os.getcwd();
    DataDir=Dir + '/DataFormatted/';
    Density=np.genfromtxt(DataDir+'DenseAll.csv');
    Depth=np.genfromtxt(DataDir+'Depth.csv');
    dz=(Depth[1]-Depth[0])*1000;
    Avocado=6.022e23; # mols to atoms conversion

    Locations=np.genfromtxt(DataDir+'SeismicLocations.csv');
    Locations[:,0]=Locations[:,0]-360;
    Qs=np.genfromtxt(DataDir+'LongLatSurfaceHeat.csv',skip_header=1,delimiter=',');
    Qm=np.genfromtxt(DataDir+'MantleHeat.txt',skip_header=1,delimiter=',');
    Temp70=Temp70km(Locations);
    
    #Qm[:,0]=Qm[:,0]+360;
    QmInterp=Nearest2D(Qm[:,0:2],Qm[:,2]*1e-3);
    QmLoc=QmInterp(Locations[:,0],Locations[:,1]);

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
    years=365.24*24*60*60;#years to seconds conversion

    UraniumHeatL=alpha238*Avocado*UContentL*LamU238/years+alpha235*Avocado*UContentL*LamU235/years/137.88;
    ThoriumHeatL=alpha232*Avocado*ThContentL*LamTh232/years;
    KHeatL=beta*Avocado*K40ContentL*LamK40/years;
    TotalHeatL=UraniumHeatL+ThoriumHeatL+KHeatL; # W/gram
    
    UraniumHeatU=alpha238*Avocado*UContentU*LamU238/years+alpha235*Avocado*UContentU*LamU235/years/137.88;
    ThoriumHeatU=alpha232*Avocado*ThContentU*LamTh232/years;
    KHeatU=beta*Avocado*K40ContentU*LamK40/years;
    TotalHeatU=UraniumHeatU+ThoriumHeatU+KHeatU;
    
    P=[-.061794, 16.3021, 8.9674];
    T=np.zeros([len(Locations),len(Depth)])+np.polyval(P,Depth);
    T=np.transpose(T);
    Fels=(3-Density)/(0.35);
    Fels[Density<2.65]=1;
    Fels[Density>3]=0;
    Mant=Density>3;
    kM=KMix(Fels,T);
    kmant=KMant(T);
    print(np.shape(Density))
    
    k=np.zeros(np.shape(kmant));
    Poro=(Density-2.65)/(1.0-2.65);
    Poro[Poro<0]=0;
    
    #k[Mant==0]=kM[Mant==0];
    k[Mant==1]=kmant[Mant==1];
    k[Density<2.65]=Poro[Density<2.65]*0.6+(1-Poro[Density<2.65])*3;
    
    TotalHeatProd=(Fels*TotalHeatU+(1-Fels)*TotalHeatL)*Density*1e6;
    Flux=1e3*np.cumsum(TotalHeatProd,0);
    Flux=Flux+QmLoc;
    
    Flux[Mant]=np.nan;
    MantInd=len(Depth)-np.sum(np.isnan(Flux),0);
    MantIndMinus=len(Depth)-MantInd;
    #print(Flux[:,0])
    #print(Depth)
    #Flux[MantIndMinus:,:]=Flux[0:MantInd,:];
    
    for i in range(np.shape(Flux)[1]):
        Flux[0:MantInd[i],i]=np.flipud(Flux[0:MantInd[i],i])
        Flux[MantInd[i]:,i]=Flux[MantInd[i],i-1];
        
    plt.close('all')
    plt.subplot(1,4,1)
    TotalHeatProd[Density>3.0]=0;
    plt.plot(TotalHeatProd*1e6,Depth,lw=0.5);
    #plt.plot(k*1e3,Depth)
    plt.ylim([0 ,70])
    plt.gca().invert_yaxis();
    plt.xlabel('Production $\mu$W/m$^3$',fontsize=10)
    plt.ylabel('Depth (km)')
    plt.yticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70])
    plt.xticks([0,.4,.8,1.2])
    
    Temp=np.zeros(np.shape(Flux));
    Temp[0,:]=10;
    for i in np.arange(1,len(Depth)):
        T[i,:]=T[i-1,:]+Flux[i-1,:]*dz/k[i-1,:];
        k[i,:]=KMix(Fels[i,:],T[i,:]);
    
    plt.subplot(1,4,2)
    plt.plot(1e3*Flux,Depth,lw=0.5)
    plt.ylim([0 ,70])
    plt.gca().invert_yaxis();
    plt.xlabel('Flux mW/m$^2$',ha='center',fontsize=10)
    plt.xticks([20,60,100,140])
    plt.yticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70],['','','','','','','','','','','','','',''])
    plt.text(100,35,'Measured at Surface',rotation=270)
    plt.text(80,35,'Need to Estimate at Mantle',rotation=270)
    
    plt.subplot(1,4,3)
    plt.plot(k,Depth)
    plt.ylim([0 ,70])
    plt.gca().invert_yaxis();
    plt.xlabel('Conductivity W/m/K$^{\circ}$',fontsize=10)
    plt.xlim([2,5])
    plt.xticks([2,3,4,5])
    plt.yticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70],['','','','','','','','','','','','','',''])
    plt.text(4,10,'Need to Compute from Inversion',rotation=270)

    plt.subplot(1,4,4)
    plt.plot(Density,Depth,c='k',lw=0.5)
    plt.ylim([0 ,70])
    plt.gca().invert_yaxis();
    plt.xlabel('Temp C$^{\circ}$',fontsize=10)
    plt.yticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70],['','','','','','','','','','','','','',''])
    #plt.xlim([0,1500])
    #plt.xticks([0,500,1000,1500])
    #plt.fill_between([np.min(Temp70),np.max(Temp70)],0,70,color='r',where=None,alpha=0.5)
    plt.text(1000,3,'V$_S$ Estimate for 70km Temp',rotation=270,color='k')

    #plt.savefig('HeatFluxProd.png');
    return k;

#k=HeatProd();
TotalHeatU=LinearSolveAll();