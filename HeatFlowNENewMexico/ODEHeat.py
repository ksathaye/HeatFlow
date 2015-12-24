# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:41:06 2015

@author: kiransathaye
"""
import numpy as np
import scipy
#import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import interp1d as interp1
import pandas

#%% sets which figures to plot
MainFig=0;
SuppFig=1;


def ComputeCost():

    df3 = pandas.DataFrame(np.random.randn(4, 5),columns=['Property', 'Water', 'Felsic', 'Mafic', 'Mantle']);
    df3['Property'][0]='A';
    df3['Property'][1]='B';
    df3['Property'][2]='Density';
    df3['Property'][3]='Heat';

    df3['Water']=[0.6,np.inf,1,0];
    df3['Felsic']=[.64,807,2.7,1.65e-6];
    df3['Mafic']=[1.18,474,3,0.19e-6];
    df3['Mantle']=[0.73,1293,3.3,0];

    print(df3)

#%% thermal conductivity of felsic upper crust
def KFels(T):
    A=0.64;
    B=807;
    k=1+A+B/(350+T);
    return k;
#%%thermal conductivity of mafic lower crust
def KMafic(T):
    A=1.18;
    B=474;
    k=1+A+B/(350+T);
    return k;
#%%averaging function for middle crust
def KMix(X,T):
    KF=KFels(T);
    KM=KMafic(T);
    k=KF*X+(1-X)*KM;
    return k;


#%% Begin script for geotherm caclculation
years=365.24*24*60*60;#years to seconds conversion

Density=np.genfromtxt('CenterDense.csv',delimiter=',');#load density from USArray seismic profile

#Load helium diffusion profiles for plot
He0=np.genfromtxt('He0Percent.csv',delimiter=',');
He1=np.genfromtxt('He1Percent.csv',delimiter=',');
FracLost=1-np.sum(He1[:,1])/np.sum(He0[:,1]);

#Load Argon diffusion profiles for plot
Ar0=np.genfromtxt('Ar0Percent.csv',delimiter=',');
Ar1=np.genfromtxt('Ar1Percent.csv',delimiter=',');

qs=58e-3;#surface heat flux in W/square meter in NE New Mexico
Density=Density[::2,:]; # smooth density data
dz=(Density[1::,0]-Density[0:-1,0])*1e3; #dz from raw density data
PoroSurface=(Density[:,1]-2.65)/(-1.65); # compute porosity in sediment column from density
Ksurface=3*(1-PoroSurface)+0.6*PoroSurface; #compute conductivity in sediment column based on mixture of qtz and water
Avocado=6.022e23; # mols to atoms conversion

UFraction=1.35e-6; #upper crust uranium mass fraction
ThFraction=UFraction*3.9; #upper crust thorium mass fraction
K2OFraction=13913*UFraction; #upper crust K2O mass fraction

dz=dz[Density[0:-1,0]<=45]; #only consider crust, remove mantle
Density=Density[Density[:,0]<=45,:];

X=(-Density[:,1]+3.0)/0.3; #
X[X<0]=0;
X[X>1]=1;

UContent=(UFraction*X+(1-X)*0.2e-6)*(Density[:,1]*1e6)/238; #W/SM of each cell
ThContent=(ThFraction*X+(1-X)*1.2e-6)*(Density[:,1]*1e6)/232;
K40Content=120e-6*(K2OFraction*X+(1-X)*0.6e-2)*Density[:,1]*1e6/94*2;

alpha238=7.41e-12;#Joules/decay
alpha235=7.24e-12;#Joules/decay
alpha232=6.24e-12;#Joules/decay
beta=1.14e-13; #Joules/decay

if 1==1:
    MeVJ=1.60218e-13
    alpha235=46.402*MeVJ
    alpha238=51.698*MeVJ
    alpha232=42.652*MeVJ
    beta=(.893*1.311+.107*1.505)*MeVJ

LamU238 = np.log(2)/(4.468*1e9);#% decay rate of U in years
LamTh232 = np.log(2)/(1.405e10); # decay rate of Th in years
LamU235 = np.log(2)/(703800000); #decay rate of 235U in years
LamK40=np.log(2)/1.248e9;#decay rate of K40 in years

H238=LamU238*alpha238*Avocado*UContent/years; # Heat from Uranium 238
H235=LamU235*alpha235*Avocado*UContent/years/137.88;# Heat from Uranium 235
H232=LamTh232*alpha232*Avocado*ThContent/years; # Heat from Thorium 232
H40=K40Content*LamK40*beta*Avocado/years;# Heat from  Potassium 40

TotalH=H238+H232+H235+H40; # total heat production at present with depth

T=np.zeros(len(TotalH)); # Initialize temp vector
k=np.zeros(len(TotalH)); #initialize conductivity vector
q=np.zeros(len(TotalH));# initialize heat flux vector
q=0.0582-np.cumsum(dz*TotalH); # compute estimated mantle heat flow
T[0]=10; #surface temperature average
k[0]=(1-PoroSurface[0])*3+PoroSurface[0]*0.6; # set conductivity at surface
Sed=np.where(PoroSurface>0); # array where sedimentary column exists

for i in range(np.max(Sed)): #start solving for temperature downward through sediment column
    k[i+1]=(1-PoroSurface[i])*KMix(X[i+1],T[i])+(PoroSurface[i])*0.6;
    T[i+1]=T[i]+q[i]*dz[i]/k[i+1];

#solve for temperature through crustal basement
for i in range(np.max(Sed),len(T)-1):
    k[i+1]=KMix(X[i],T[i]);
    T[i+1]=T[i]+q[i]*dz[i]/k[i+1];

z=Density[:,0]; #depth vector
P=np.polyfit(z,T,2); # polynomial quadratic aporoximation of temperature profile (coefficents)
TP=np.polyval(P,z); #vector of polynomial approximation

print(q[-1]*1e3)

#%% Plotting Figure 1
if MainFig==1:

    """
    plt.subplot(1,5,3) # plot temperature profile
    plt.plot(T,Density[:,0],lw=2);# plot temperature
    plt.gca().invert_yaxis();#invert y axis for depth downward
    plt.title('Temp',fontsize=9) #plot title
    plt.xlabel('Celsius',fontsize=9); #plot units on X axis
    plt.xticks([0,200,400,600],fontsize=8)# set up x ticks
    plt.xlim([0,650]);# set up x limits
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[]);# yticks and remove labels


    plt.subplot(1,2,1) #  plot density as a function of depth
    plt.plot(Density[:,1],Density[:,0],lw=2)
    plt.gca().invert_yaxis();#invert y axis for depth downward
    plt.title('Bulk Density',fontsize=12);#plot title
    plt.xticks([2.2,2.6,3],fontsize=12);
    plt.yticks([0,5,10,15,20,25,30,35,40,45],fontsize=12);
    plt.text(2.9,4,'A',fontsize=16)
    plt.xlim([2.1,3.1]);# set up x limits
    plt.xlabel('g/cc',fontsize=12);#plot units on X axis
    plt.ylabel('Depth (km)',fontsize=12)

    plt.subplot(1,5,2) # Plot total heat production as function of depth
    plt.plot(TotalH*1e6,Density[:,0],lw=2);#plot total heat production with depth
    plt.gca().invert_yaxis(); #invert y axis for depth downward
    plt.title('Heat',fontsize=9);#title on plot
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[])# yticks and remove labels
    plt.xticks([.2,.6,1],fontsize=8);# place xticks and labels
    plt.xlabel('$\mu$W/m$^3$',fontsize=9);# xlabel with units
    """
    plt.close('all')
    plt.figure();
    plt.subplot(1,2,1) # plot helium production and diffusion profile
    plt.plot(He1[:,1]*1e3,He1[:,0]*1e-3,c='red',lw=2)
    plt.plot(He0[:,1]*1e3,He0[:,0]*1e-3,c='b',lw=2)
    blue_line = mlines.Line2D([], [], color='blue',label='$\phi$=0',lw=2)
    red_line = mlines.Line2D([], [], color='red',label='$\phi$=1%',lw=2)
    #plt.text(10,23,'$\phi$=1%',color='red',fontsize=14);#label on plot
    #plt.text(10,20,'$\phi$=0%',color='b',fontsize=14);#label on plot
    plt.xlabel('10$^{-3}$ mol/m$^3$')
    plt.title('$^4$He, 1.5Ga')
    plt.xticks([0,16,32,48,64],fontsize=12)
    plt.xlim([-1,64])
    #plt.yticks([0,5,10,15,20,25,30,35,40,45])
    #plt.text(60,4,'B',fontsize=16)
    plt.gca().invert_yaxis();
    lg=plt.legend(handles=[blue_line,red_line],loc=1,fontsize=12);# place legedn on figure
    plt.ylabel('Depth (km)',fontsize=12)
    plt.text(58,43,'A',fontsize=18);
    plt.grid();

    plt.subplot(1,2,2) #plot Argon diffusion profiles
    plt.plot(Ar1[:,1]*1e3,Ar1[:,0]*1e-3,c='red',lw=2); #plot 1% porosity  profile
    plt.plot(Ar0[:,1]*1e3,Ar0[:,0]*1e-3,c='b',lw=2);#plot 0% porosity  profile
    blue_line = mlines.Line2D([], [], color='blue',label='$\phi$=0',lw=2);#legend for 0 diffusion
    red_line = mlines.Line2D([], [], color='red',label='$\phi$=1%',lw=2);#legend for 1% porosity
    #plt.text(2,23,'$\phi$=1%',color='red',fontsize=14);#label on plot
    #plt.text(2,20,'$\phi$=0%',color='b',fontsize=14);#label on plot
    plt.xlabel('10$^{-3}$ mol/m$^3$');#xlabel with units
    plt.title('$^{40}$Ar, 1.5Ga');# title
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[])#remove yticklabels
    plt.xticks([0,3,6,9,12,15],fontsize=12);#place xticks and labels
    plt.xlim([-1,15])
    plt.text(11.5,43,'B',fontsize=18);
    lg=plt.legend(handles=[blue_line,red_line],loc=1,fontsize=12);# place legend on figure
    plt.gca().invert_yaxis();
    plt.grid();

    plt.savefig('DiffProfiles.pdf',format='pdf', dpi=1000) #save figure to PDF

#%% Plotting Figure 2

if SuppFig==1:
    plt.figure();

    plt.subplot(1,5,5); #plot temperature with depth
    plt.plot(T,Density[:,0],lw=2); #plot temperature
    plt.gca().invert_yaxis();##invert y axis
    plt.title('Temperature',fontsize=10); #plot title
    plt.xlabel('Celsius',fontsize=12); # xlabel with units
    plt.ylabel('Depth (km)'); # ylabel for depth
    plt.xticks([0,200,400,600],fontsize=9);# x tick marks
    plt.xlim([0,650]); # x limits
    plt.grid();#plot grid
    plt.text(50,43,'E',fontsize=14)
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[]);# remove y tick labels


    plt.subplot(1,5,4);#plots thermal conductivity with depth
    plt.plot(k,Density[:,0],lw=2);# plot thermal conductivity
    plt.gca().invert_yaxis();#invert y axis
    plt.title('Conductivity',fontsize=9); #plot title
    plt.xlabel('W/(m$^{\circ}$K)');#plot units on xlabel
    plt.xticks([2,3,4],fontsize=9); # plot xticks
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[]);# add y tick labels
    plt.grid();# plot grid
    plt.text(2.2,43,'D',fontsize=14)

    plt.subplot(1,5,1); #plots heat flux  with depth
    plt.plot(q*1e3,Density[:,0],lw=2);#plots heat flux  with depth
    plt.gca().invert_yaxis();#invert y axis
    plt.title('Heat Flux',fontsize=9); # plot title
    plt.xlabel('mW/m$^2$');#xlabel with units
    #plt.xticks([28,38,48,58],fontsize=9);#set xticks for readability
    plt.xticks([22,31,40,49,58],fontsize=9);#set xticks for readability
    plt.xlim([22,60]); # set x limits
    plt.text(52,43,'A',fontsize=14)
    plt.yticks([0,5,10,15,20,25,30,35,40,45]);#add y tick labels
    plt.grid();#plot grid

    plt.subplot(1,5,3);#plot heat production from each element
    plt.plot((H238+H235)*1e6,Density[:,0],lw=2);#U235 and U238 heat
    plt.plot(H232*1e6,Density[:,0],lw=2,c='red',ls='-');#Thorium 232 Heat
    plt.plot(H40*1e6,Density[:,0],lw=2,c='green');# Potassium heat
    plt.gca().invert_yaxis(); #invert y axis
    plt.title('Heat Production',fontsize=9); # plot title
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[])#remove yticks after 1st plot
    plt.xticks([0,.2,.4],fontsize=9);#set xticks for readability
    plt.grid();#plot grid
    plt.text(0.32,43,'C',fontsize=14)
    blue_line = mlines.Line2D([], [], color='blue',label='\n',lw=2);#legend entry for heat from Uranium
    red_line = mlines.Line2D([], [], color='red',label='\n',lw=2,ls='-');# legend entry for  heat from Thorium
    green_line = mlines.Line2D([], [], color='green',label='\n',lw=2);# legend entry for heat from Potassium
    lg=plt.legend(handles=[blue_line,red_line,green_line],loc=2,fontsize=10);# place legedn on figure
    lg.draw_frame(False) #remove legend box
    plt.text(.11,3.5,'U',color='b',fontsize=14)
    plt.text(.085,8.8,'Th',color='r',fontsize=14)
    plt.text(.11,14,'K',color='g',fontsize=14)
    plt.xlim([0,0.45])
    plt.xlabel('$\mu$W/m$^3$',fontsize=12); #x axis units for heat productoin

    plt.subplot(1,5,2);#plot heat production from each element
    plt.plot(Density[:,1],Density[:,0],lw=2);#U235 and U238 heat
    plt.gca().invert_yaxis(); #invert y axis
    plt.title('Bulk Density',fontsize=9); # plot title
    plt.yticks([0,5,10,15,20,25,30,35,40,45],[])#remove yticks after 1st plot
    plt.xticks([2.2,2.6,3],fontsize=9);#set xticks for readability
    plt.xlabel('g/cc',fontsize=12); #x axis units for heat production
    plt.grid();#plot grid
    plt.text(2.3,43,'B',fontsize=14)

    plt.savefig('SuppGeoTherm.pdf',format='pdf'); # save figure to file