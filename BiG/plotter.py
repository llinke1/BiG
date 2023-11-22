import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colorbar import ColorbarBase

from helpers_plot import initPlot, finalizePlot

class plotter:
    def __init__ (self, cmap='plasma', **kwargs) -> None:
        initPlot(**kwargs)

        self.cmap=cmap
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

    
    def plotBispectrum3D(self, outputFn="", showplot=True, tightlayout=False, names=[], vmin=3, vmax=20):
        if len(names)==0:
            names=list(self.ks.keys())

        N=len(names)

        if N>1:
            fig, axes=plt.subplots(ncols=N, subplot_kw={'projection': '3d'}, figsize=(N*5+5, 6))

            for i, name in enumerate(names):
                axes[i].set_title(name)
                img=axes[i].scatter(self.ks[name][:,0], 
                            self.ks[name][:,1], 
                            self.ks[name][:,2], 
                            c=np.log(self.bispecs[name]), cmap=self.cmap, vmin=vmin, vmax=vmax)
                axes[i].set_xlabel(r'$k_1$ [$h$/Mpc]')
                axes[i].set_ylabel(r'$k_2$ [$h$/Mpc]')
                axes[i].set_zlabel(r'$k_3$ [$h$/Mpc]')
            fig.colorbar(img, ax=axes.ravel(), label=r'$\ln[B(k_1, k_2, k_3)]$', orientation='vertical')
        else:
            name=names[0]
            fig, axes=plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7,6))
            axes.set_title(name)
            img=axes.scatter(self.ks[name][:,0], 
                            self.ks[name][:,1], 
                            self.ks[name][:,2], 
                            c=np.log(self.bispecs[name]), cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes.set_xlabel(r'$k_1$ [$h$/Mpc]')
            axes.set_ylabel(r'$k_2$ [$h$/Mpc]')
            axes.set_zlabel(r'$k_3$ [$h$/Mpc]')
            fig.colorbar(img, ax=axes, label=r'$\ln[B(k_1, k_2, k_3)]$', orientation='vertical')
        
        
        finalizePlot(axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)

    def plotBispectrum1D(self, k1_k3=0, k2_k3=0, mode="", outputFn="", showplot=True, tightlayout=False, names=[], colors=[], markers=[]):
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

        if len(colors)==0:
            cmap=plt.get_cmap(self.cmap)
            for i in range(N):
                a=(i+1)/(N+1)
                c=cmap(a)
                colors.append(c)


        fig,ax=plt.subplots()
        ax.loglog()

        for i,name in enumerate(names):
            mask=(self.ratio_edges_low[name][:,0]<k1_k3) \
            &(self.ratio_edges_high[name][:,0]>k1_k3) \
            & (self.ratio_edges_low[name][:,1]<k2_k3) \
            & (self.ratio_edges_high[name][:,1]>k2_k3)

            k=self.ks[name][:,2][mask]
            bispec=self.bispecs[name][mask]
            if len(markers)==0:
                ax.plot(k, bispec, label=name, c=colors[i])
            else:
                ax.plot(k, bispec, label=name, ls=':', marker=markers[i], color=colors[i])
        
        #y=0.5*(k2_k3**2-k1_k3**2+1)
        y=np.sqrt(k2_k3**2-0.25*(k2_k3**2+1-k1_k3**2)**2)
        x=np.sqrt(k1_k3**2-y**2)
        triangle=Polygon([[0,0], [1,0], [x,y]], fc='none', ec='k')
        line=Polygon([[0,0], [1,0]], fc='none', ec='r', lw=3)


        ax.set_xlabel(r'$k_3$ [$h$/Mpc]')
        ax.set_ylabel(r'$B($'+f"{k1_k3}"+"$\,k_3,$"+f"{k2_k3}"+r"$\,k_3, k_3)$")
        ax.legend()
        
        ax2=fig.add_axes([0.2,0.2,0.2,0.2])
        ax2.set_axis_off()
        ax2.add_patch(triangle)
        ax2.add_patch(line)


        finalizePlot(fig.get_axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)
        

    def plotBispectrumTriangle(self, k3, outputFn="", showplot=True, tightlayout=False, names=[], vmin=0, vmax=1):
        if len(names)==0:
            names=list(self.ks.keys())
        N=len(names)

        if N>1:
            fig, axes=plt.subplots(ncols=N, figsize=(N*5+2, 5))
            for i, name in enumerate(names):

                axes[i].set_title(name+r", $k_3=$"+f"{k3}"+ r"[$h$/Mpc]")

                mask=(self.kbinedges_low[name][:,2]<k3) \
                &(self.kbinedges_high[name][:,2]>k3)

                img=axes[i].scatter(self.ratio_13[name][mask], self.ratio_23[name][mask], c=np.log(self.bispecs[name][mask]), cmap=self.cmap, vmin=vmin, vmax=vmax)

                axes[i].set_xlabel(r'$k_1/k_3$')
                axes[i].set_ylabel(r'$k_2/k_3$')
            fig.colorbar(img, ax=axes.ravel(), label=r'$\ln[B(k_1, k_2, k_3)]$', orientation='vertical') 
        else:
            name=names[0]
            fig, axes=plt.subplots(figsize=(7,5))
            axes.set_title(name+r", $k_3=$"+f"{k3}"+ r"[$h$/Mpc]")
            mask=(self.kbinedges_low[name][:,2]<k3) \
                &(self.kbinedges_high[name][:,2]>k3)

            img=axes.scatter(self.ratio_13[name][mask], self.ratio_23[name][mask], c=np.log(self.bispecs[name][mask]),cmap=self.cmap, vmin=vmin, vmax=vmax)


            axes.set_xlabel(r'$k_1/k_3$')
            axes.set_ylabel(r'$k_2/k_3$')
            fig.colorbar(img, ax=axes, label=r'$\ln[B(k_1, k_2, k_3)]$', orientation='vertical')
        

        finalizePlot(axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)


    def plotBispectrumTriangleBinsize(self, k3, outputFn="", showplot=True, tightlayout=False, names=[], vmin=0, vmax=1):
        if len(names)==0:
            names=list(self.ks.keys())
        N=len(names)

        if N>1:
            fig, axes=plt.subplots(ncols=N, figsize=(N*5+2, 5))
            for i, name in enumerate(names):

                axes[i].set_title(name+r", $k_3=$"+f"{k3}"+ r" $h$/Mpc")

                mask=(self.kbinedges_low[name][:,2]<k3) \
                &(self.kbinedges_high[name][:,2]>k3)

                r13=self.ratio_13[name][mask]
                r23=self.ratio_23[name][mask]
                bispec=np.log(self.bispecs[name][mask])
                r13_lower=self.ratio_edges_low[name][mask][:,0]
                r13_upper=self.ratio_edges_high[name][mask][:,0]
                r23_lower=self.ratio_edges_low[name][mask][:,1]
                r23_upper=self.ratio_edges_high[name][mask][:,1]

                cmap=plt.get_cmap('plasma')
                norm=plt.Normalize(vmin, vmax)
                c=[cmap(norm(d)) for d in bispec]

                axes[i].bar(x=0.5*(r13_upper+r13_lower), height=r23_upper-r23_lower, bottom=r23_lower, width=r13_upper-r13_lower, color=c)
                axes[i].scatter(r13.ravel(), r23.ravel(), color='k', marker='x', zorder=10)

                axes[i].set_xlabel(r'$k_1/k_3$')
                axes[i].set_ylabel(r'$k_2/k_3$')
            cbar = ColorbarBase(ax=fig.add_axes([0.9, 0.1, 0.02, 0.8]), cmap=cmap, norm=norm)
            cbar.set_label(r'$\ln [B(k_1, k_2, k_3)]$')
        else:
            name=names[0]
            fig, axes=plt.subplots(figsize=(7,5))
            axes.set_title(name+r", $k_3=$"+f"{k3}"+ r" $h$/Mpc")
            mask=(self.kbinedges_low[name][:,2]<k3) \
                &(self.kbinedges_high[name][:,2]>k3)

            r13=self.ratio_13[name][mask]
            r23=self.ratio_23[name][mask]
            bispec=np.log(self.bispecs[name][mask])
            r13_lower=self.ratio_edges_low[name][mask][:,0]
            r13_upper=self.ratio_edges_high[name][mask][:,0]
            r23_lower=self.ratio_edges_low[name][mask][:,1]
            r23_upper=self.ratio_edges_high[name][mask][:,1]

            cmap=plt.get_cmap('plasma')
            norm=plt.Normalize(vmin, vmax)
            c=[cmap(norm(d)) for d in bispec]

            axes.bar(x=0.5*(r13_upper+r13_lower), height=r23_upper-r23_lower, bottom=r23_lower, width=r13_upper-r13_lower, color=c)
            axes.scatter(r13.ravel(), r23.ravel(), color='k', marker='x', zorder=10)

            axes.set_xlabel(r'$k_1/k_3$')
            axes.set_ylabel(r'$k_2/k_3$')
            cbar = ColorbarBase(ax=fig.add_axes([0.9, 0.1, 0.02, 0.8]), cmap=cmap, norm=norm)
            cbar.set_label(r'$\ln [B(k_1, k_2, k_3)]$')
        

        finalizePlot(axes, outputFn=outputFn, showplot=showplot, tightlayout=tightlayout, showlegend=False)
