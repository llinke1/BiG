import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from helpers_plot import initPlot, finalizePlot

class plotter:
    def __init__ (self, **kwargs) -> None:
        initPlot(**kwargs)

        self.ks={}
        self.bispecs={}

        self.kbinedges_low={}
        self.kbinedges_high={}

        self.ratio_13={}
        self.ratio_23={}
        self.ratio_edges_low={}
        self.ratio_edges_high={}

    
    def addBispectrum(self, ks, kbinedges_low, kbinedges_high, bispec, name=""):
        self.ks[name]=ks
        self.kbinedges_low[name]=kbinedges_low
        self.kbinedges_high[name]=kbinedges_high
        self.bispecs[name]=bispec

        ratio_13=ks[:,0]/ks[:,2]
        self.ratio_13[name]=ratio_13
        ratio_23=ks[:,1]/ks[:,2]
        self.ratio_23[name]=ratio_23

        ratio_edges_low=kbinedges_low.copy()
        ratio_edges_low[:,0]=kbinedges_low[:,0]/ks[:,2]
        ratio_edges_low[:,1]=kbinedges_low[:,1]/ks[:,2]
        self.ratio_edges_low[name]=ratio_edges_low

        ratio_edges_high=kbinedges_high.copy()
        ratio_edges_high[:,0]=kbinedges_high[:,0]/ks[:,2]
        ratio_edges_high[:,1]=kbinedges_high[:,1]/ks[:,2]
        self.ratio_edges_high[name]=ratio_edges_high

    
    def plotBispectrum3D(self, outputFn="", showplot=True, tightlayout=False, names=[], vmin=0, vmax=1):
        if len(names)==0:
            names=list(self.ks.keys())

        N=len(names)


        if N>1:
            fig, axes=plt.subplots(ncols=N, subplot_kw={'projection': '3d'}, figsize=(N*5+2, 5))

            for i, name in enumerate(names):
                #axes[i]=fig.add_subplot(projection='3d')
                axes[i].set_title(name)
                img=axes[i].scatter(self.ks[name][:,0], 
                            self.ks[name][:,1], 
                            self.ks[name][:,2], 
                            self.bispecs[name], vmin=vmin, vmax=vmax)
                axes[i].set_xlabel(r'$k_1$ [$h$/Mpc]')
                axes[i].set_ylabel(r'$k_2$ [$h$/Mpc]')
                axes[i].set_zlabel(r'$k_3$ [$h$/Mpc]')
            fig.colorbar(img, ax=axes.ravel(), label=r'$B(k_1, k_2, k_3)$', orientation='vertical')
        else:
            name=names[0]
            fig, axes=plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7,5))
            axes.set_title(name)
            img=axes.scatter(self.ks[name][:,0], 
                            self.ks[name][:,1], 
                            self.ks[name][:,2], 
                            self.bispecs[name], vmin=vmin, vmax=vmax)
            axes.set_xlabel(r'$k_1$ [$h$/Mpc]')
            axes.set_ylabel(r'$k_2$ [$h$/Mpc]')
            axes.set_zlabel(r'$k_3$ [$h$/Mpc]')
            fig.colorbar(img, ax=axes, label=r'$B(k_1, k_2, k_3)$', orientation='vertical')
        
        
        finalizePlot(axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)

    def plotBispectrum1D(self, k1_k3=0, k2_k3=0, mode="", outputFn="", showplot=True, tightlayout=True, names=[]):
        if len(names)==0:
            names=list(self.ks.keys())

        if mode=="equilateral":
            k1_k3=1
            k2_k3=1
        elif mode=="folded":
            k1_k3=0.5
            k2_k3=0.5
        elif mode=="squeezed":
            k1_k3=0
            k2_k3=1

        N=len(names)



        fig,ax=plt.subplots()
        ax.loglog()

        for name in names:
            mask=(self.ratio_edges_low[name][:,0]<k1_k3) \
            &(self.ratio_edges_high[name][:,0]>k1_k3) \
            & (self.ratio_edges_low[name][:,1]<k2_k3) \
            & (self.ratio_edges_high[name][:,1]>k2_k3)

            k=self.ks[name][:,2][mask]
            bispec=self.bispecs[name][mask]
   
            ax.plot(k, bispec, label=name)
        
        y=0.5*(k1_k3**2-k2_k3**2+1)
        x=np.sqrt(k1_k3**2-y**2)
        triangle=Polygon([[0,0], [1,0], [x,y]], fc='none', ec='k')

        ax.set_xlabel(r'$k_3$ [$h$/Mpc]')
        ax.set_ylabel(r'$B($'+f"{k1_k3}"+"$\,k_3,$"+f"{k2_k3}"+r"$\,k_3, k_3)$")
        ax.legend()
        
        ax2=fig.add_axes([0.2,0.2,0.2,0.2])
        ax2.set_axis_off()
        ax2.add_patch(triangle)

        finalizePlot(fig.get_axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)
        

    def plotBispectrumTriangle(self, k3, outputFn="", showplot=True, tightlayout=False, names=[], vmin=0, vmax=1):
        if len(names)==0:
            names=list(self.ks.keys())
        N=len(names)
        if N>1:
            fig, axes=plt.subplots(ncols=N, figsize=(N*5+5, 5))
            for i,name in enumerate(names):
                
                axes[i].set_title(name+r", $k_3=$"+f"{k3}"+ r"\,$h$/Mpc")

                mask=(self.kbinedges_low[name][:,2]<k3) \
                &(self.kbinedges_high[name][:,2]>k3)

                img=axes[i].scatter(self.ratio_13[name][mask], self.ratio_23[name][mask], c=self.bispecs[name][mask], vmin=vmin, vmax=vmax)

                axes[i].set_xlabel(r'$k_1/k_3$')
                axes[i].set_ylabel(r'$k_2/k_3$')


            fig.colorbar(img, ax=axes.ravel(), label=r'$B(k_1, k_2, k_3)$')
        else:
            fig, axes=plt.subplots(figsize=(7,5))
            name=names[0] 
            axes.set_title(name+r", $k_3=$"+f"{k3}"+ r"\,$h$/Mpc")

            mask=(self.kbinedges_low[name][:,2]<k3) \
                &(self.kbinedges_high[name][:,2]>k3)

            img=axes.scatter(self.ratio_13[name][mask], self.ratio_23[name][mask], c=self.bispecs[name][mask], vmin=vmin, vmax=vmax)
            axes.set_xlabel(r'$k_1/k_3$')
            axes.set_ylabel(r'$k_2/k_3$')

            fig.colorbar(img, ax=axes, label=r'$B(k_1, k_2, k_3)$')

        finalizePlot(fig.get_axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)
